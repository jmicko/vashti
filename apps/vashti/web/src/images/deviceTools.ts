import { requestBlob, requestJson } from "../api";
import type { PrivateChatAttachment, PrivateChatMessage } from "../privateChatStore";
import type { Recipe, Preview } from "./types";

export type DeviceImageContext = {
  messages: () => PrivateChatMessage[];
  preview: (frame: Preview | null) => void;
  save: (attachment: PrivateChatAttachment) => Promise<void>;
};
const active = new Map<string, Set<AbortController>>();
export function cancelDeviceImages(chatId: string) {
  for (const controller of active.get(chatId) ?? []) controller.abort();
}
export async function executeDeviceImageTool(name: string, args: Record<string, unknown>, chatId: string, messageId: string, context: DeviceImageContext) {
  const controller = new AbortController();
  const calls = active.get(chatId) ?? new Set<AbortController>();
  active.set(chatId, calls); calls.add(controller);
  let job: string | null = null;
  try {
    const messages = context.messages();
    const message = messages.find(m => m.id === messageId && m.chat_id === chatId);
    if (!message || message.status === "stopped") throw new Error("The private image request was stopped");
    const attachments = new Map<string, PrivateChatAttachment>();
    const visited = new Set<string>();
    let current: PrivateChatMessage | undefined = message;
    while (current && !visited.has(current.id)) {
      visited.add(current.id);
      if (!current.is_deleted) for (const attachment of current.attachments) {
        if (!attachment.revision_id || attachment.revision_id === current.active_revision_id) attachments.set(attachment.id, attachment);
      }
      current = messages.find(m => m.id === current?.parent_message_id && m.chat_id === chatId);
    }
    const edit = name === "edit_image";
    const ids = args.image_ids ?? [];
    if (!Array.isArray(ids) || ids.length > 10 || (edit && !ids.length) || (!edit && ids.length)) throw new Error("Choose up to ten images from this conversation for editing");
    const references = ids.map(id => {
      const attachment = attachments.get(String(id));
      const data = attachment?.data_url?.match(/^data:image\/(?:png|jpeg|webp);base64,(.+)$/s)?.[1];
      if (!data) throw new Error("An input image is not available in this conversation branch");
      return data;
    });
    if (typeof args.prompt !== "string") throw new Error("An image prompt is required");
    const started = await requestJson<{ id: string }>("/api/images/private", { method: "POST", body: JSON.stringify({ operation:edit ? "edit" : "generate", prompt:args.prompt, settings:args.settings ?? {}, reference_images:references }), signal:controller.signal });
    job = started.id;
    let status: { status: string; preview?: Preview | null; recipe: Recipe; ignored_settings: string[] };
    while (true) {
      if (controller.signal.aborted) throw new Error("Image generation stopped");
      status = await requestJson(`/api/images/private/${job}`, { signal:controller.signal });
      context.preview(status.preview ?? null);
      if (status.status === "complete") break;
      if (status.status === "failed") throw new Error("Kuno could not generate this private image. Check model readiness and image settings.");
      await new Promise(resolve => window.setTimeout(resolve, 1000));
    }
    const bytes = await requestBlob(`/api/images/private/${job}/result`, { signal:controller.signal });
    const dataUrl = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(new Error("Could not read private image")); reader.readAsDataURL(bytes); });
    if (controller.signal.aborted) throw new Error("Image generation stopped");
    const id = crypto.randomUUID();
    await context.save({ id, chat_id:chatId, message_id:messageId, revision_id:message.active_revision_id, original_filename:"generated-image.png", mime_type:"image/png", size_bytes:bytes.size, attachment_kind:"image", created_at:Math.floor(Date.now()/1000), data_url:dataUrl, image_recipe:status.recipe });
    return { status:"complete", image_id:id, width:status.recipe.width, height:status.recipe.height, seed:status.recipe.seed, ignored_settings:status.ignored_settings };
  } finally {
    context.preview(null);
    if (job) void requestJson(`/api/images/private/${job}/stop`, { method:"POST" }).catch(() => undefined);
    calls.delete(controller); if (!calls.size) active.delete(chatId);
  }
}
