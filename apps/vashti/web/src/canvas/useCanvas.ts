import { lazy, useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";
import { splitThinkingDelta } from "../chatMessages";
import { canvasReferences } from "./references";
import { canvasChanged } from "./repository";
import type { CanvasMode, CanvasReference } from "./types";
export const LazyCanvasPanel = lazy(() =>
  import("./Workspace").then((module) => ({
    default: module.CanvasWorkspacePanel,
  })),
);
export type CanvasOpenRequest = {
  key: number;
  copy?: { document_id: string; title: string; content: string };
  reference?: CanvasReference;
};
export function useChatCanvas(
  chat: string,
  mode: CanvasMode,
  onOpen: () => void,
  isActive: (id: string) => boolean,
) {
  const [request, setRequest] = useState<CanvasOpenRequest | null>(null),
    sequence = useRef(0),
    opened = useRef(new Set<string>());
  useEffect(() => {
    setRequest(null);
    opened.current.clear();
  }, [chat, mode]);
  const copyMessage = useCallback(
    (message: ChatMessage) => {
      setRequest({
        key: ++sequence.current,
        copy: {
          document_id: crypto.randomUUID(),
          title:
            message.role === "assistant" ? "Response draft" : "Message draft",
          content: message.active_revision?.content_text ?? "",
        },
      });
      onOpen();
    },
    [onOpen],
  );
  const openReference = useCallback(
    (reference: CanvasReference) => {
      setRequest({ key: ++sequence.current, reference });
      onOpen();
    },
    [onOpen],
  );
  function notice(id: string, delta: string) {
    if (!delta.includes("canvas")) return;
    const refs = canvasReferences(splitThinkingDelta(delta));
    if (!refs.length) return;
    canvasChanged(chat, mode);
    if (delta.includes('"create_canvas"') && !opened.current.has(id)) {
      opened.current.add(id);
      if (isActive(id)) {
        setRequest({
          key: ++sequence.current,
          reference: refs[refs.length - 1],
        });
        onOpen();
      }
    }
  }
  return {
    request,
    copyMessage,
    openReference,
    notice,
    consumed: (key: number) =>
      setRequest((current) => (current?.key === key ? null : current)),
  };
}
