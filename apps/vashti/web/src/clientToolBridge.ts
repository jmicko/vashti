import { requestJson } from "./api";
import {
  DEVICE_NOTE_TOOL_NAMES,
  executeDeviceNoteTool,
  type DeviceNoteToolName
} from "./deviceNoteTools";
import {
  getPrivateClientToolReceipt,
  savePrivateClientToolReceipt,
  unixTimestamp
} from "./privateChatStore";

export type ClientToolCallEvent = {
  type: "client_tool_call";
  assistant_message_id: string;
  generation_id: string;
  call_id: string;
  resume_token: string;
  name: string;
  arguments: Record<string, unknown>;
};

export type ClientToolExecutionContext = {
  chatId: string;
  modelKey: string;
  modelName: string;
};

type ClientToolHandler = (
  args: Record<string, unknown>,
  event: ClientToolCallEvent,
  context: ClientToolExecutionContext
) => Promise<unknown>;

const handlers = new Map<string, ClientToolHandler>();
const inFlightCalls = new Map<string, Promise<void>>();

for (const name of DEVICE_NOTE_TOOL_NAMES) {
  handlers.set(name, (args, event, context) =>
    executeDeviceNoteTool(name as DeviceNoteToolName, args, {
      chatId: context.chatId,
      messageId: event.assistant_message_id,
      callId: event.call_id,
      modelKey: context.modelKey,
      modelName: context.modelName
    })
  );
}

export async function dispatchClientToolCall(
  event: ClientToolCallEvent,
  context: ClientToolExecutionContext
): Promise<void> {
  const existingCall = inFlightCalls.get(event.call_id);
  if (existingCall) {
    return existingCall;
  }

  const call = executeAndDeliver(event, context).finally(() => {
    inFlightCalls.delete(event.call_id);
  });
  inFlightCalls.set(event.call_id, call);
  return call;
}

async function executeAndDeliver(
  event: ClientToolCallEvent,
  context: ClientToolExecutionContext
): Promise<void> {
  let existing;
  try {
    existing = await getPrivateClientToolReceipt(event.call_id);
  } catch (receiptError) {
    await deliverResult(
      event,
      undefined,
      receiptError instanceof Error
        ? `Device tool storage is unavailable: ${receiptError.message}`
        : "Device tool storage is unavailable"
    );
    return;
  }
  if (existing?.generation_id === event.generation_id) {
    await deliverResult(event, existing.result, existing.error);
    return;
  }
  if (existing) {
    await deliverResult(event, undefined, "Device tool call identity conflict");
    return;
  }

  const handler = handlers.get(event.name);
  let result: unknown;
  let error: string | undefined;
  try {
    if (!handler) {
      throw new Error(`No device handler is registered for ${event.name}`);
    }
    result = await handler(event.arguments ?? {}, event, context);
  } catch (executionError) {
    error = executionError instanceof Error ? executionError.message : "Device tool failed";
  }

  try {
    await savePrivateClientToolReceipt({
      id: event.call_id,
      generation_id: event.generation_id,
      result,
      error,
      created_at: unixTimestamp()
    });
  } catch {
    // Returning the result prevents a completed local mutation from leaving the
    // generation blocked. The server also deduplicates completed call IDs.
  }
  await deliverResult(event, result, error);
}

async function deliverResult(
  event: ClientToolCallEvent,
  result: unknown,
  error: string | undefined
) {
  let lastError: unknown;
  for (const delay of [0, 250, 750, 1_500]) {
    if (delay > 0) {
      await new Promise((resolve) => window.setTimeout(resolve, delay));
    }
    try {
      await requestJson("/api/private/client-tool-result", {
        method: "POST",
        body: JSON.stringify({
          generation_id: event.generation_id,
          call_id: event.call_id,
          resume_token: event.resume_token,
          ...(error === undefined ? { result } : { error })
        })
      });
      return;
    } catch (requestError) {
      lastError = requestError;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Failed to return device tool result");
}
