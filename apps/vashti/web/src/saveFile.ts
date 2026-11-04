import { invokeNative, isNativeRuntime } from "./runtime";

export type FileSaveOutcome = "saved" | "cancelled" | "requested";

export async function saveFile(blob: Blob, name: string, mimeType = blob.type): Promise<FileSaveOutcome> {
  if (blob.size > 20 * 1024 * 1024) throw new Error("File exceeds the 20 MiB download limit");
  const filename = name.replace(/[^a-zA-Z0-9_. -]/g, "_").slice(0, 160);
  if (!filename.trim().replace(/\./g, "")) throw new Error("Invalid filename");
  if (isNativeRuntime()) {
    const data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1]);
      reader.onerror = () => reject(new Error("Could not read file"));
      reader.readAsDataURL(blob);
    });
    const result = await invokeNative<{ saved: boolean }>("native_save_file", {
      filename, mimeType: mimeType || "application/octet-stream", data
    });
    return result.saved ? "saved" : "cancelled";
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return "requested";
}
