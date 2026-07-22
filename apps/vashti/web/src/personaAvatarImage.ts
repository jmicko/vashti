const JPEG_MIME = "image/jpeg";
const PNG_MIME = "image/png";
const GIF_MIME = "image/gif";

export async function normalizePersonaImageFile(file: File): Promise<File> {
  // Some mobile browsers expose picker files through a short-lived content-provider
  // handle. Materialize the bytes while that handle is valid instead of retaining it
  // until the user eventually saves the form.
  const bytes = new Uint8Array(await file.arrayBuffer());
  const detectedType = detectPreservedType(bytes);
  if (detectedType) {
    return new File([bytes], file.name || defaultFilename(detectedType), {
      type: detectedType,
      lastModified: file.lastModified
    });
  }

  const ownedFile = new File([bytes], file.name || "profile-image", {
    type: file.type,
    lastModified: file.lastModified
  });
  const image = await loadImage(ownedFile);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("This browser could not convert the image");
  }
  context.drawImage(image, 0, 0);
  const blob = await canvasBlob(canvas, PNG_MIME);
  return new File([blob], pngFilename(file.name), {
    type: PNG_MIME,
    lastModified: file.lastModified
  });
}

export const normalizePersonaAvatarFile = normalizePersonaImageFile;

function detectPreservedType(bytes: Uint8Array) {
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return PNG_MIME;
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return JPEG_MIME;
  }
  if (
    bytes.length >= 6 &&
    bytes[0] === 0x47 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x38 &&
    (bytes[4] === 0x37 || bytes[4] === 0x39) &&
    bytes[5] === 0x61
  ) {
    return GIF_MIME;
  }
  return null;
}

function loadImage(file: File) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      if (image.naturalWidth === 0 || image.naturalHeight === 0) {
        reject(new Error("The image has no readable dimensions"));
        return;
      }
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This image format could not be read by the browser"));
    };
    image.src = url;
  });
}

function canvasBlob(canvas: HTMLCanvasElement, type: string) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("The image could not be converted"));
      }
    }, type);
  });
}

function pngFilename(filename: string) {
  const base = filename.replace(/\.[^./\\]+$/, "").trim() || "profile-image";
  return `${base}.png`;
}

function defaultFilename(type: string) {
  if (type === JPEG_MIME) {
    return "profile-image.jpg";
  }
  if (type === GIF_MIME) {
    return "profile-image.gif";
  }
  return "profile-image.png";
}
