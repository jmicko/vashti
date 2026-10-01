import { requestJson } from "../api";
export const imagePath = "/api/images";
export const writeImage = <T>(url: string, value: unknown, method = "POST") =>
  requestJson<T>(url, { method, body: JSON.stringify(value) });
export const imageError = (error: unknown) =>
  error instanceof Error ? error.message : "Image operation failed";
export function notifyImageFlows() {
  window.dispatchEvent(new Event("vashti:image-flows-changed"));
}
