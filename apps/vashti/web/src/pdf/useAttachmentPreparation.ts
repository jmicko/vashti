import { useEffect, useRef } from "react";
import type { ComposerAttachment } from "../types";
import type { PdfPreparation } from "./types";

export type PrepareAttachment = (file: File, options?: PdfPreparation) => Promise<ComposerAttachment> | ComposerAttachment;

export function useAttachmentPreparation() {
  const inFlight = useRef(new Map<string, AbortController>());
  const batch = useRef(0);
  useEffect(() => {
    const pending = inFlight.current;
    return () => {
      batch.current++;
      for (const task of pending.values()) task.abort();
    };
  }, []);

  return {
    cancel(id: string) { inFlight.current.get(id)?.abort(); },
    cancelAll() {
      batch.current++;
      for (const task of inFlight.current.values()) task.abort();
    },
    trackBatch() {
      const current = batch.current;
      return () => current === batch.current;
    },
    async prepare(
      id: string, file: File, upload: PrepareAttachment,
      onProgress: (label: string) => void,
      discardUpload?: (file: ComposerAttachment) => Promise<void>
    ) {
      const task = new AbortController();
      inFlight.current.set(id, task);
      try {
        const attachment = await upload(file, {
          signal: task.signal,
          onProgress: label => { if (!task.signal.aborted) onProgress(label); }
        });
        if (!task.signal.aborted) return attachment;
        if (attachment.status === "uploaded" && !attachment.isExisting) {
          await discardUpload?.(attachment).catch(() => undefined);
        }
      } catch (cause) {
        if (!task.signal.aborted) throw cause;
      } finally {
        inFlight.current.delete(id);
      }
    }
  };
}
