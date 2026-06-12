import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent
} from "react";
import { ImagePlus, RotateCcw, Trash2 } from "lucide-react";
import {
  DEFAULT_AVATAR_CROP,
  avatarCropRect,
  clamp,
  normalizeAvatarCrop,
  type AvatarCrop,
  type CropRect,
  type ImageDimensions
} from "./avatarCrop";
import { useModelAvatarSource } from "./ModelAvatar";
import { normalizePersonaAvatarFile } from "./personaAvatarImage";

type CropHandle = "north-west" | "north-east" | "south-west" | "south-east";

type CropLayout = {
  imageLeft: number;
  imageTop: number;
  imageWidth: number;
  imageHeight: number;
  scale: number;
  cropLeft: number;
  cropTop: number;
  cropSize: number;
};

type DragState = {
  pointerId: number;
  mode: "move" | "resize";
  handle?: CropHandle;
  startClientX: number;
  startClientY: number;
  startCrop: AvatarCrop;
  startRect: CropRect;
};

type LoadedImage = ImageDimensions & {
  src: string;
};

export function PersonaAvatarField({
  displayName,
  assetId,
  privateAssetId,
  previewFile,
  cropX,
  cropY,
  cropSize,
  onFileChange,
  onRemove,
  onCropChange
}: {
  displayName: string;
  assetId: string | null;
  privateAssetId: string | null;
  previewFile: File | null;
  cropX: number;
  cropY: number;
  cropSize: number;
  onFileChange: (file: File) => void;
  onRemove: () => void;
  onCropChange: (cropX: number, cropY: number, cropSize: number) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [loadedImage, setLoadedImage] = useState<LoadedImage | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [fileError, setFileError] = useState<string | null>(null);
  const hasImage = Boolean(assetId || privateAssetId || previewFile);
  const src = useModelAvatarSource({ assetId, privateAssetId, previewFile });
  const dimensions =
    src && loadedImage?.src === src
      ? { width: loadedImage.width, height: loadedImage.height }
      : null;
  const crop = useMemo(
    () => normalizeAvatarCrop({ x: cropX, y: cropY, size: cropSize }, dimensions),
    [cropSize, cropX, cropY, dimensions]
  );

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    const updateSize = () =>
      setStageSize({
        width: stage.clientWidth,
        height: stage.clientHeight
      });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [hasImage]);

  const layout = useMemo(
    () => cropLayout(stageSize, dimensions, crop),
    [crop, dimensions, stageSize]
  );

  const updateCrop = useCallback(
    (nextCrop: AvatarCrop) => {
      const normalized = normalizeAvatarCrop(nextCrop, dimensions);
      onCropChange(normalized.x, normalized.y, normalized.size);
    },
    [dimensions, onCropChange]
  );

  function startPointerDrag(
    event: ReactPointerEvent<HTMLElement>,
    mode: DragState["mode"],
    handle?: CropHandle
  ) {
    if (!dimensions || !layout) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    stageRef.current?.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      mode,
      handle,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startCrop: crop,
      startRect: avatarCropRect(dimensions, crop)
    };
  }

  function movePointer(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !dimensions || !layout) {
      return;
    }
    event.preventDefault();
    const deltaX = (event.clientX - drag.startClientX) / layout.scale;
    const deltaY = (event.clientY - drag.startClientY) / layout.scale;

    if (drag.mode === "move") {
      updateCrop({
        ...drag.startCrop,
        x: ((drag.startRect.centerX + deltaX) / dimensions.width) * 100,
        y: ((drag.startRect.centerY + deltaY) / dimensions.height) * 100
      });
      return;
    }

    const handle = drag.handle;
    if (!handle) {
      return;
    }
    const sizeDelta = resizeProjection(handle, deltaX, deltaY);
    const fixedPoint = oppositeCorner(handle, drag.startRect);
    const minimumSize = Math.min(dimensions.width, dimensions.height) * 0.1;
    const maximumSize = maximumCropSize(handle, fixedPoint, dimensions);
    const nextSize = clamp(drag.startRect.size + sizeDelta, minimumSize, maximumSize);
    const center = cropCenterFromFixedCorner(handle, fixedPoint, nextSize);
    updateCrop({
      x: (center.x / dimensions.width) * 100,
      y: (center.y / dimensions.height) * 100,
      size: (nextSize / Math.min(dimensions.width, dimensions.height)) * 100
    });
  }

  function endPointer(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    if (stageRef.current?.hasPointerCapture(event.pointerId)) {
      stageRef.current.releasePointerCapture(event.pointerId);
    }
  }

  async function chooseFile(file: File) {
    setFileError(null);
    try {
      onFileChange(await normalizePersonaAvatarFile(file));
    } catch (error) {
      setFileError(error instanceof Error ? error.message : "Could not read profile image");
    }
  }

  return (
    <section className="persona-avatar-field" aria-label="Custom model profile image">
      <div
        ref={stageRef}
        className="persona-avatar-crop-stage"
        onPointerMove={movePointer}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}
        onLostPointerCapture={endPointer}
      >
        {src && hasImage ? (
          <>
            <img
              className="persona-avatar-crop-source"
              src={src}
              alt=""
              draggable={false}
              onLoad={(event) =>
                setLoadedImage({
                  src,
                  width: event.currentTarget.naturalWidth,
                  height: event.currentTarget.naturalHeight
                })
              }
              onError={() => setFileError("Could not display this profile image")}
            />
            {layout && (
              <>
                <div
                  className="persona-avatar-circle-preview"
                  style={{
                    left: layout.cropLeft,
                    top: layout.cropTop,
                    width: layout.cropSize,
                    height: layout.cropSize
                  }}
                  aria-hidden="true"
                >
                  <img
                    src={src}
                    alt=""
                    draggable={false}
                    style={{
                      left: layout.imageLeft - layout.cropLeft,
                      top: layout.imageTop - layout.cropTop,
                      width: layout.imageWidth,
                      height: layout.imageHeight
                    }}
                  />
                </div>
                <div
                  className="persona-avatar-crop-selection"
                  style={{
                    left: layout.cropLeft,
                    top: layout.cropTop,
                    width: layout.cropSize,
                    height: layout.cropSize
                  }}
                  role="application"
                  aria-label="Profile image crop. Drag to reposition."
                  onPointerDown={(event) => startPointerDrag(event, "move")}
                >
                  {(["north-west", "north-east", "south-west", "south-east"] as const).map(
                    (handle) => (
                      <span
                        key={handle}
                        className={`persona-avatar-crop-handle persona-avatar-crop-handle-${handle}`}
                        aria-hidden="true"
                        onPointerDown={(event) => startPointerDrag(event, "resize", handle)}
                      />
                    )
                  )}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="persona-avatar-empty">
            <span>{displayName.trim().charAt(0).toLocaleUpperCase() || "?"}</span>
            <p>Choose an image to set a profile crop.</p>
          </div>
        )}
      </div>
      <div className="persona-avatar-controls">
        <div className="persona-avatar-actions">
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.currentTarget.value = "";
              if (file) {
                void chooseFile(file);
              }
            }}
          />
          <button type="button" className="secondary-button" onClick={() => inputRef.current?.click()}>
            <ImagePlus />
            <span>{hasImage ? "Replace Image" : "Choose Image"}</span>
          </button>
          {hasImage && (
            <>
              <button
                type="button"
                className="icon-button"
                title="Reset crop"
                aria-label="Reset profile image crop"
                onClick={() =>
                  onCropChange(
                    DEFAULT_AVATAR_CROP.x,
                    DEFAULT_AVATAR_CROP.y,
                    DEFAULT_AVATAR_CROP.size
                  )
                }
              >
                <RotateCcw />
              </button>
              <button type="button" className="secondary-button danger-button" onClick={onRemove}>
                <Trash2 />
                <span>Remove</span>
              </button>
            </>
          )}
        </div>
        {fileError && <p className="form-error">{fileError}</p>}
        <p>
          Drag the square to reposition it, or drag a corner to resize it. The clear circle is what
          appears in the app.
        </p>
        <p>JPEG, PNG, and GIF originals are preserved. Other readable images are saved as PNG.</p>
      </div>
    </section>
  );
}

function cropLayout(
  stage: { width: number; height: number },
  dimensions: ImageDimensions | null,
  crop: AvatarCrop
): CropLayout | null {
  if (!dimensions || stage.width <= 0 || stage.height <= 0) {
    return null;
  }
  const scale = Math.min(stage.width / dimensions.width, stage.height / dimensions.height);
  const imageWidth = dimensions.width * scale;
  const imageHeight = dimensions.height * scale;
  const imageLeft = (stage.width - imageWidth) / 2;
  const imageTop = (stage.height - imageHeight) / 2;
  const rect = avatarCropRect(dimensions, crop);

  return {
    imageLeft,
    imageTop,
    imageWidth,
    imageHeight,
    scale,
    cropLeft: imageLeft + rect.left * scale,
    cropTop: imageTop + rect.top * scale,
    cropSize: rect.size * scale
  };
}

function resizeProjection(handle: CropHandle, deltaX: number, deltaY: number) {
  switch (handle) {
    case "north-west":
      return (-deltaX - deltaY) / 2;
    case "north-east":
      return (deltaX - deltaY) / 2;
    case "south-west":
      return (-deltaX + deltaY) / 2;
    case "south-east":
      return (deltaX + deltaY) / 2;
  }
}

function oppositeCorner(handle: CropHandle, rect: CropRect) {
  switch (handle) {
    case "north-west":
      return { x: rect.left + rect.size, y: rect.top + rect.size };
    case "north-east":
      return { x: rect.left, y: rect.top + rect.size };
    case "south-west":
      return { x: rect.left + rect.size, y: rect.top };
    case "south-east":
      return { x: rect.left, y: rect.top };
  }
}

function maximumCropSize(
  handle: CropHandle,
  fixed: { x: number; y: number },
  dimensions: ImageDimensions
) {
  switch (handle) {
    case "north-west":
      return Math.min(fixed.x, fixed.y);
    case "north-east":
      return Math.min(dimensions.width - fixed.x, fixed.y);
    case "south-west":
      return Math.min(fixed.x, dimensions.height - fixed.y);
    case "south-east":
      return Math.min(dimensions.width - fixed.x, dimensions.height - fixed.y);
  }
}

function cropCenterFromFixedCorner(
  handle: CropHandle,
  fixed: { x: number; y: number },
  size: number
) {
  switch (handle) {
    case "north-west":
      return { x: fixed.x - size / 2, y: fixed.y - size / 2 };
    case "north-east":
      return { x: fixed.x + size / 2, y: fixed.y - size / 2 };
    case "south-west":
      return { x: fixed.x - size / 2, y: fixed.y + size / 2 };
    case "south-east":
      return { x: fixed.x + size / 2, y: fixed.y + size / 2 };
  }
}
