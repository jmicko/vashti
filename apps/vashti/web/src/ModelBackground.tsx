import type { CSSProperties } from "react";
import { Image as ImageIcon } from "lucide-react";
import { useModelAvatarSource } from "./ModelAvatar";
import type { ModelBackgroundMode, ModelBackgroundSettings } from "./types";

type BackgroundLayout = {
  mode: ModelBackgroundMode;
  x: number;
  y: number;
  scale: number;
};

export function ModelBackgroundLayer({
  background
}: {
  background: ModelBackgroundSettings | null;
}) {
  const src = useModelAvatarSource({
    assetId: background?.background_is_private ? null : background?.background_asset_id,
    privateAssetId: background?.background_is_private ? background?.background_asset_id : null
  });
  if (!background?.background_asset_id || !src) return null;
  return (
    <div
      className="model-background-layer"
      aria-hidden="true"
      style={backgroundLayerStyle(src, background)}
    />
  );
}

export function modelBackgroundContainerStyle(
  background: ModelBackgroundSettings | null
): CSSProperties | undefined {
  if (!background?.background_asset_id) return undefined;
  return {
    "--model-background-message-dim": background.background_message_dim
  } as CSSProperties;
}

export function ModelBackgroundPreview({
  assetId,
  privateAssetId,
  previewFile,
  layout,
  dim,
  messageDim,
  orientation
}: {
  assetId: string | null;
  privateAssetId?: string | null;
  previewFile: File | null;
  layout: BackgroundLayout;
  dim: number;
  messageDim: number;
  orientation: "landscape" | "portrait";
}) {
  const src = useModelAvatarSource({
    assetId: privateAssetId ? null : assetId,
    privateAssetId,
    previewFile
  });
  const style = src
    ? {
        backgroundImage: `url("${src}")`,
        ...backgroundImageProperties(layout)
      }
    : undefined;

  return (
    <div
      className={`model-background-preview model-background-preview-${orientation}`}
      style={style}
    >
      <div className="model-background-preview-dim" style={{ opacity: dim }} />
      <div
        className="model-background-preview-copy"
        style={{ background: `rgba(var(--background-chat-panel-rgb), ${messageDim})` }}
      >
        <span>MODEL</span>
        <p>The background stays visible while messages remain readable.</p>
      </div>
    </div>
  );
}

export function ModelBackgroundButton({
  assetId,
  privateAssetId,
  previewFile,
  label,
  onClick
}: {
  assetId?: string | null;
  privateAssetId?: string | null;
  previewFile?: File | null;
  label: string;
  onClick: () => void;
}) {
  const src = useModelAvatarSource({ assetId, privateAssetId, previewFile });
  return (
    <button
      type="button"
      className={`model-background-button${src ? " model-background-button-preview" : ""}`}
      aria-label={label}
      title={label}
      onClick={onClick}
      style={src ? { backgroundImage: `url("${src}")` } : undefined}
    >
      <span className="model-background-button-shade" />
      <ImageIcon />
    </button>
  );
}

function backgroundLayerStyle(src: string, background: ModelBackgroundSettings) {
  const landscape = backgroundImageProperties({
    mode: background.background_landscape_mode,
    x: background.background_landscape_x,
    y: background.background_landscape_y,
    scale: background.background_landscape_scale
  });
  const portrait = backgroundImageProperties({
    mode: background.background_portrait_mode,
    x: background.background_portrait_x,
    y: background.background_portrait_y,
    scale: background.background_portrait_scale
  });

  return {
    backgroundImage: `url("${src}")`,
    "--model-background-dim": background.background_dim,
    "--model-background-landscape-size": landscape.backgroundSize,
    "--model-background-landscape-position": landscape.backgroundPosition,
    "--model-background-landscape-repeat": landscape.backgroundRepeat,
    "--model-background-portrait-size": portrait.backgroundSize,
    "--model-background-portrait-position": portrait.backgroundPosition,
    "--model-background-portrait-repeat": portrait.backgroundRepeat
  } as CSSProperties;
}

function backgroundImageProperties(layout: BackgroundLayout): CSSProperties {
  const backgroundSize =
    layout.mode === "fill"
      ? "cover"
      : layout.mode === "fit"
        ? "contain"
        : layout.mode === "stretch"
          ? "100% 100%"
          : `${layout.scale}% auto`;

  return {
    backgroundSize,
    backgroundPosition: `${layout.x}% ${layout.y}%`,
    backgroundRepeat: layout.mode === "tile" ? "repeat" : "no-repeat"
  };
}
