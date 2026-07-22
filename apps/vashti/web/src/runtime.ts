const NATIVE_ASSET_ORIGIN = "http://vashtiasset.localhost";

let nativeAssetNamespace = "unselected";

export function isNativeRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function invokeNative<T>(
  command: string,
  args: Record<string, unknown> = {}
): Promise<T> {
  if (!isNativeRuntime()) {
    throw new Error("Native runtime is unavailable");
  }

  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export function setNativeAssetNamespace(namespace: string | null) {
  nativeAssetNamespace = namespace?.trim() || "unselected";
}

export function apiAssetUrl(path: string) {
  if (!isNativeRuntime()) {
    return path;
  }

  return `${NATIVE_ASSET_ORIGIN}/${encodeURIComponent(nativeAssetNamespace)}/${encodePath(path)}`;
}

function encodePath(path: string) {
  const bytes = new TextEncoder().encode(path);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/gu, "-").replace(/\//gu, "_").replace(/=+$/u, "");
}
