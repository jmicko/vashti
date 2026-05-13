import { responseErrorMessage } from "./api";
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
