import { useEffect, useState } from "react";
import { Save, X } from "lucide-react";
import { RetroLoader } from "./common";
import { ModelAvatar } from "./ModelAvatar";
import { PersonaAvatarField } from "./PersonaAvatarField";

export type ModelAvatarEdit = {
  assetId: string | null;
  file: File | null;
  cropX: number;
  cropY: number;
  cropSize: number;
};

export function ModelAvatarEditorDialog({
  displayName,
  title,
  assetId,
  cropX,
  cropY,
  cropSize,
  inheritedAvatar,
  isBusy,
  error,
  onCancel,
  onSave
}: {
  displayName: string;
  title: string;
  assetId: string | null;
  cropX: number;
  cropY: number;
  cropSize: number;
  inheritedAvatar?: {
    assetId: string;
    cropX: number;
    cropY: number;
    cropSize: number;
  } | null;
  isBusy: boolean;
  error: string | null;
  onCancel: () => void;
  onSave: (edit: ModelAvatarEdit) => void;
}) {
  const [nextAssetId, setNextAssetId] = useState(assetId);
  const [file, setFile] = useState<File | null>(null);
  const [nextCropX, setNextCropX] = useState(cropX);
  const [nextCropY, setNextCropY] = useState(cropY);
  const [nextCropSize, setNextCropSize] = useState(cropSize);
  const [isReadingFile, setIsReadingFile] = useState(false);
  const controlsDisabled = isBusy || isReadingFile;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !controlsDisabled) {
        onCancel();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [controlsDisabled, onCancel]);

  return (
    <div
      className="confirm-backdrop model-avatar-editor-backdrop"
      role="presentation"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget && !controlsDisabled) {
          onCancel();
        }
      }}
    >
      <section
        className="model-avatar-editor-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-avatar-editor-title"
      >
        <div className="model-avatar-editor-header">
          <div>
            <p className="eyebrow">Profile Image</p>
            <h2 id="model-avatar-editor-title">{title}</h2>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Close profile image editor"
            disabled={controlsDisabled}
            onClick={onCancel}
          >
            <X />
          </button>
        </div>
        {!assetId && inheritedAvatar && !file && (
          <div className="model-avatar-inherited">
            <ModelAvatar
              displayName={displayName}
              assetId={inheritedAvatar.assetId}
              cropX={inheritedAvatar.cropX}
              cropY={inheritedAvatar.cropY}
              cropSize={inheritedAvatar.cropSize}
            />
            <p>
              Using the server default. Choose an image to create a personal override.
            </p>
          </div>
        )}
        <PersonaAvatarField
          displayName={displayName}
          assetId={nextAssetId}
          privateAssetId={null}
          previewFile={file}
          cropX={nextCropX}
          cropY={nextCropY}
          cropSize={nextCropSize}
          onFileChange={(nextFile) => setFile(nextFile)}
          onReadingChange={setIsReadingFile}
          onRemove={() => {
            setNextAssetId(null);
            setFile(null);
            setNextCropX(50);
            setNextCropY(50);
            setNextCropSize(100);
          }}
          onCropChange={(x, y, size) => {
            setNextCropX(x);
            setNextCropY(y);
            setNextCropSize(size);
          }}
        />
        {error && <p className="error">{error}</p>}
        <div className="dialog-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={controlsDisabled}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={controlsDisabled}
            onClick={() =>
              onSave({
                assetId: nextAssetId,
                file,
                cropX: nextCropX,
                cropY: nextCropY,
                cropSize: nextCropSize
              })
            }
          >
            {controlsDisabled ? <RetroLoader /> : <Save />}
            <span>{isReadingFile ? "Reading image..." : isBusy ? "Saving..." : "Save"}</span>
          </button>
        </div>
      </section>
    </div>
  );
}
