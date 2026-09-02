import { memo, useEffect, useMemo, useState } from "react";
import { avatarImageStyle } from "./avatarCrop";
import {
  getCachedPrivatePersonaAvatar,
  getPrivatePersonaAvatar
} from "./privateChatStore";
import { useDecodedModelMedia } from "./modelMediaCache";
import { apiAssetUrl } from "./runtime";

export const ModelAvatar = memo(function ModelAvatar({
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
  const media = useDecodedModelMedia(src, "avatar");
  const classes = ["model-avatar", className].filter(Boolean).join(" ");

  if (!src || !media.ready || media.failed) {
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
        loading="eager"
        draggable={false}
        style={avatarImageStyle(media.dimensions, { x: cropX, y: cropY, size: cropSize })}
      />
    </span>
  );
});

export function useModelAvatarSource({
  assetId,
  privateAssetId,
  previewFile
}: {
  assetId?: string | null;
  privateAssetId?: string | null;
  previewFile?: File | null;
}) {
  const [privateUrl, setPrivateUrl] = useState<string | null>(() =>
    privateAssetId ? getCachedPrivatePersonaAvatar(privateAssetId)?.data_url ?? null : null
  );
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
    setPrivateUrl(
      privateAssetId ? getCachedPrivatePersonaAvatar(privateAssetId)?.data_url ?? null : null
    );
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
