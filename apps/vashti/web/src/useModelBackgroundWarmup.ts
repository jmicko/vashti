import { useEffect } from "react";
import { warmDecodedModelMedia } from "./modelMediaCache";
import { getPrivatePersonaAvatar, type PrivatePersona } from "./privateChatStore";
import { apiAssetUrl } from "./runtime";
import type {
  BackendModelGroup,
  ModelBackgroundSettings,
  ModelInfo,
  Persona
} from "./types";

const BACKGROUND_WARM_LIMIT = 10;

type Candidate = {
  assetId: string;
  isPrivate: boolean;
};

type NetworkInformation = {
  saveData?: boolean;
  effectiveType?: string;
};

type IdleWindow = {
  requestIdleCallback?: (
    callback: () => void,
    options?: { timeout: number }
  ) => number;
  cancelIdleCallback?: (handle: number) => void;
};

export function useModelBackgroundWarmup({
  groups,
  personas,
  privatePersonas,
  selectedModelInfo
}: {
  groups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  selectedModelInfo: ModelInfo | null;
}) {
  useEffect(() => {
    if (document.visibilityState !== "visible") return;

    const connection = (navigator as Navigator & { connection?: NetworkInformation }).connection;
    if (
      connection?.saveData ||
      connection?.effectiveType === "slow-2g" ||
      connection?.effectiveType === "2g"
    ) {
      return;
    }

    const candidates = backgroundCandidates(
      groups,
      personas,
      privatePersonas,
      selectedModelInfo
    );
    if (candidates.length === 0) return;

    let cancelled = false;
    const warm = async () => {
      for (const candidate of candidates) {
        if (cancelled) return;
        try {
          const src = candidate.isPrivate
            ? (await getPrivatePersonaAvatar(candidate.assetId))?.data_url ?? null
            : apiAssetUrl(
                `/api/persona-avatars/${encodeURIComponent(candidate.assetId)}`
              );
          if (src) await warmDecodedModelMedia(src, "background");
        } catch {
          // Preloading is opportunistic; normal rendering can retry later.
        }
      }
    };

    const idleWindow = window as unknown as IdleWindow;
    const handle = idleWindow.requestIdleCallback
      ? idleWindow.requestIdleCallback(() => void warm(), { timeout: 3_000 })
      : window.setTimeout(() => void warm(), 1_200);

    return () => {
      cancelled = true;
      if (idleWindow.requestIdleCallback && idleWindow.cancelIdleCallback) {
        idleWindow.cancelIdleCallback(handle);
      } else {
        window.clearTimeout(handle);
      }
    };
  }, [groups, personas, privatePersonas, selectedModelInfo]);
}

function backgroundCandidates(
  groups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[],
  selectedModelInfo: ModelInfo | null
) {
  const candidates: Candidate[] = [];
  const seen = new Set<string>();
  const add = (background: ModelBackgroundSettings | null | undefined, isPrivate = false) => {
    const assetId = background?.background_asset_id;
    if (!assetId) return;
    const privateAsset = background?.background_is_private ?? isPrivate;
    const key = `${privateAsset ? "private" : "hosted"}:${assetId}`;
    if (seen.has(key) || candidates.length >= BACKGROUND_WARM_LIMIT) return;
    seen.add(key);
    candidates.push({ assetId, isPrivate: privateAsset });
  };

  add(selectedModelInfo);
  for (const group of groups) {
    for (const model of group.models) {
      if (model.is_favorite) add(model);
    }
  }
  for (const persona of personas) {
    if (persona.is_favorite) add(persona.current_version);
  }
  for (const persona of privatePersonas) {
    if (persona.is_favorite) add(persona.current_version, true);
  }
  for (const group of groups) {
    for (const model of group.models) add(model);
  }
  for (const persona of personas) add(persona.current_version);
  for (const persona of privatePersonas) add(persona.current_version, true);

  return candidates;
}
