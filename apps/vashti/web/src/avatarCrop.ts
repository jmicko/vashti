import type { CSSProperties } from "react";

export const DEFAULT_AVATAR_CROP = {
  x: 50,
  y: 50,
  size: 100
};

export type AvatarCrop = typeof DEFAULT_AVATAR_CROP;

export type ImageDimensions = {
  width: number;
  height: number;
};

export type CropRect = {
  left: number;
  top: number;
  size: number;
  centerX: number;
  centerY: number;
};

export function normalizeAvatarCrop(
  crop: Partial<AvatarCrop>,
  dimensions?: ImageDimensions | null
): AvatarCrop {
  const size = clamp(finiteOr(crop.size, 100), 10, 100);
  const x = clamp(finiteOr(crop.x, 50), 0, 100);
  const y = clamp(finiteOr(crop.y, 50), 0, 100);

  if (!dimensions || dimensions.width <= 0 || dimensions.height <= 0) {
    return { x, y, size };
  }

  const sourceSize = Math.min(dimensions.width, dimensions.height) * (size / 100);
  const halfX = (sourceSize / dimensions.width) * 50;
  const halfY = (sourceSize / dimensions.height) * 50;
  return {
    x: clamp(x, halfX, 100 - halfX),
    y: clamp(y, halfY, 100 - halfY),
    size
  };
}

export function avatarCropRect(
  dimensions: ImageDimensions,
  crop: Partial<AvatarCrop>
): CropRect {
  const normalized = normalizeAvatarCrop(crop, dimensions);
  const size = Math.min(dimensions.width, dimensions.height) * (normalized.size / 100);
  const centerX = dimensions.width * (normalized.x / 100);
  const centerY = dimensions.height * (normalized.y / 100);

  return {
    left: centerX - size / 2,
    top: centerY - size / 2,
    size,
    centerX,
    centerY
  };
}

export function avatarImageStyle(
  dimensions: ImageDimensions | null,
  crop: Partial<AvatarCrop>
): CSSProperties {
  if (!dimensions) {
    return {
      objectFit: "cover",
      objectPosition: `${finiteOr(crop.x, 50)}% ${finiteOr(crop.y, 50)}%`
    };
  }

  const rect = avatarCropRect(dimensions, crop);
  return {
    position: "absolute",
    width: `${(dimensions.width / rect.size) * 100}%`,
    height: `${(dimensions.height / rect.size) * 100}%`,
    maxWidth: "none",
    left: `${-(rect.left / rect.size) * 100}%`,
    top: `${-(rect.top / rect.size) * 100}%`
  };
}

function finiteOr(value: number | undefined, fallback: number) {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

export function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}
