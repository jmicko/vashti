import { responseErrorMessage } from "./api";
import { invokeNative, isNativeRuntime } from "./runtime";
import type { GenerateEvent } from "./types";

export async function readGenerateEventStream({
  path,
  body,
  signal,
  onEvent
}: {
  path: string;
  body: unknown;
  signal: AbortSignal;
  onEvent: (event: GenerateEvent) => void;
}) {
  if (isNativeRuntime()) {
    await readNativeGenerateEventStream({ path, body, signal, onEvent });
    return;
  }

  const response = await fetch(path, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal
  });

  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }

  if (!response.body) {
    throw new Error("Generation stream was empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        onEvent(JSON.parse(trimmed) as GenerateEvent);
      }
    }
  }

  buffer += decoder.decode();
  const trailing = buffer.trim();
  if (trailing) {
    onEvent(JSON.parse(trailing) as GenerateEvent);
  }
}

async function readNativeGenerateEventStream({
  path,
  body,
  signal,
  onEvent
}: {
  path: string;
  body: unknown;
  signal: AbortSignal;
  onEvent: (event: GenerateEvent) => void;
}) {
  const { Channel } = await import("@tauri-apps/api/core");
  const requestId = crypto.randomUUID();
  const parser = createEventParser(onEvent);
  const channel = new Channel<{ chunk_base64: string }>((event) =>
    parser.pushBase64(event.chunk_base64)
  );
  const cancel = () => {
    void invokeNative("native_cancel_request", { requestId }).catch(() => undefined);
  };

  if (signal.aborted) {
    throw new DOMException("Generation cancelled", "AbortError");
  }

  signal.addEventListener("abort", cancel, { once: true });
  try {
    await invokeNative("native_http_stream", {
      request: {
        request_id: requestId,
        method: "POST",
        path,
        headers: { "content-type": "application/json" },
        body_text: JSON.stringify(body)
      },
      onEvent: channel
    });
    parser.finish();
  } catch (error) {
    if (signal.aborted) {
      throw new DOMException("Generation cancelled", "AbortError");
    }
    throw error;
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}

function createEventParser(onEvent: (event: GenerateEvent) => void) {
  const decoder = new TextDecoder();
  let buffer = "";

  function consumeLines() {
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        onEvent(JSON.parse(trimmed) as GenerateEvent);
      }
    }
  }

  return {
    pushBase64(chunkBase64: string) {
      const binary = atob(chunkBase64);
      const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
      buffer += decoder.decode(bytes, { stream: true });
      consumeLines();
    },
    finish() {
      buffer += decoder.decode();
      const trailing = buffer.trim();
      if (trailing) {
        onEvent(JSON.parse(trailing) as GenerateEvent);
      }
    }
  };
}
