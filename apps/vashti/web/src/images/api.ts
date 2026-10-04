import { requestJson as serverJson, requestMultipartJson as serverUpload } from "../api";
import { deviceImagePath } from "./storage";
export const imagePath = "/api/images";
export async function requestJson<T = unknown>(url: string, options: RequestInit = {}): Promise<T> {
  if (url.startsWith(`${deviceImagePath}/`)) {
    const { deviceImageRequest } = await import("./deviceRepository");
    return deviceImageRequest<T>(url.slice(deviceImagePath.length), options);
  }
  return serverJson<T>(url, options);
}
export async function requestMultipartJson<T>(url: string, body: FormData, options: RequestInit = {}): Promise<T> {
  if (url.startsWith(`${deviceImagePath}/`)) {
    const { uploadDeviceImage } = await import("./deviceRepository");
    const file = body.get("image");
    if (!(file instanceof Blob)) throw new Error("Choose an image");
    return await uploadDeviceImage(file) as T;
  }
  return serverUpload<T>(url, body, options);
}
export const writeImage = <T>(url: string, value: unknown, method = "POST") =>
  requestJson<T>(url, { method, body: JSON.stringify(value) });
export const imageError = (error: unknown) =>
  error instanceof Error ? error.message : "Image operation failed";
export function notifyImageFlows() {
  window.dispatchEvent(new Event("vashti:image-flows-changed"));
}
