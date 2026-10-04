import { invokeNative, isNativeRuntime } from "./runtime";

export async function imageDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read image"));
    reader.readAsDataURL(blob);
  });
}

export async function saveImage(blob: Blob, name: string) {
  const extension = {
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/png": "png",
  }[blob.type];
  if (!extension) throw new Error("Unsupported image format");
  const filename = `${name.replace(/[^a-zA-Z0-9_-]/g, "_")}.${extension}`;
  if (isNativeRuntime()) {
    await invokeNative("native_save_image", {
      filename,
      mimeType: blob.type,
      data: (await imageDataUrl(blob)).split(",")[1],
    });
    return;
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
