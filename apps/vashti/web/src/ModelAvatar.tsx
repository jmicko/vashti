import { useEffect, useMemo, useState } from "react";
import { avatarImageStyle, type ImageDimensions } from "./avatarCrop";
import { getPrivatePersonaAvatar } from "./privateChatStore";
import { apiAssetUrl } from "./runtime";

type LoadedImage = ImageDimensions & {
  src: string;
};

export function ModelAvatar({
  displayName,
  assetId,
  privateAssetId,
  previewFile,
  cropX = 50,
  cropY = 50,
  cropSize = 100,
  className = ""
}: {
  displayName: string;
  assetId?: string | null;
  privateAssetId?: string | null;
  previewFile?: File | null;
  cropX?: number;
  cropY?: number;
  cropSize?: number;
  className?: string;
}) {
  const src = useModelAvatarSource({ assetId, privateAssetId, previewFile });
  const [loadedImage, setLoadedImage] = useState<LoadedImage | null>(null);
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const classes = ["model-avatar", className].filter(Boolean).join(" ");
  const dimensions =
    src && loadedImage?.src === src
      ? { width: loadedImage.width, height: loadedImage.height }
      : null;

  if (!src || failedSrc === src) {
    return (
      <span className={`${classes} model-avatar-fallback`} aria-hidden="true">
        {initialFor(displayName)}
      </span>
    );
  }

  return (
    <span className={classes}>
      <img
        src={src}
        alt=""
        decoding="async"
        loading="lazy"
        draggable={false}
        style={avatarImageStyle(dimensions, { x: cropX, y: cropY, size: cropSize })}
        onLoad={(event) =>
          setLoadedImage({
            src,
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight
          })
        }
        onError={() => setFailedSrc(src)}
      />
    </span>
  );
}

export function useModelAvatarSource({
  assetId,
  privateAssetId,
  previewFile
}: {
  assetId?: string | null;
  privateAssetId?: string | null;
  previewFile?: File | null;
}) {
  const [privateUrl, setPrivateUrl] = useState<string | null>(null);
  const previewUrl = useMemo(
    () => (previewFile ? URL.createObjectURL(previewFile) : null),
    [previewFile]
  );

  useEffect(
    () => () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    },
    [previewUrl]
  );

  useEffect(() => {
    let cancelled = false;
    setPrivateUrl(null);
    if (!privateAssetId) {
      return () => {
        cancelled = true;
      };
    }

    void getPrivatePersonaAvatar(privateAssetId)
      .then((asset) => {
        if (!cancelled) {
          setPrivateUrl(asset?.data_url ?? null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPrivateUrl(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [privateAssetId]);

  const src =
    previewUrl ??
    privateUrl ??
    (assetId
      ? apiAssetUrl(`/api/persona-avatars/${encodeURIComponent(assetId)}`)
      : null);
  return src;
}

function initialFor(displayName: string) {
  return displayName.trim().charAt(0).toLocaleUpperCase() || "?";
}
