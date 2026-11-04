// Avoid Uint8Array.from(string, callback), which allocates an intermediate array
// and calls JS once per byte for multi-megabyte images on mobile.
export function imageBlobFromData(data: string): Blob {
  const comma = data.indexOf(",");
  const mime = /^data:(image\/[^;]+);base64$/.exec(data.slice(0, comma))?.[1];
  if (!mime || comma < 0) throw new Error("Image could not be read");
  const binary = atob(data.slice(comma + 1));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++)
    bytes[index] = binary.charCodeAt(index);
  return new Blob([bytes], { type: mime });
}

export async function imageThumbnail(blob: Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  try {
    const side = Math.min(bitmap.width, bitmap.height);
    const size = Math.min(320, side);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Thumbnail renderer unavailable");
    context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2,
      side, side, 0, 0, size, size);
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob(
      value => value ? resolve(value) : reject(new Error("Thumbnail unavailable")), "image/png",
    ));
  } finally {
    bitmap.close();
  }
}
