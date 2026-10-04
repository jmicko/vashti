import { createContext, useContext, type ReactNode } from "react";

export type ImageStorage = "server" | "device";
export const deviceImagePath = "/device/images";
const StorageContext = createContext<ImageStorage>("server");
export function ImageStorageProvider({
  mode,
  children,
}: {
  mode: ImageStorage;
  children: ReactNode;
}) {
  return (
    <StorageContext.Provider value={mode}>{children}</StorageContext.Provider>
  );
}
export function useImagePath() {
  return useContext(StorageContext) === "device"
    ? deviceImagePath
    : "/api/images";
}
