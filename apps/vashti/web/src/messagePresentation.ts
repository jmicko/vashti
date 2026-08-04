import type { ChatMessage, CustomModelType } from "./types";

type PersonaVersionClassification = {
  id: string;
  model_type?: CustomModelType;
};

type PersonaLinkedMessage = Pick<
  ChatMessage,
  "id" | "role" | "active_child_message_id" | "persona_version_id"
>;

export function characterPersonaVersionIds(
  versions: readonly PersonaVersionClassification[]
): Set<string> {
  return new Set(
    versions.filter((version) => version.model_type === "character").map((version) => version.id)
  );
}

export function characterPresentationMessageIds<Message extends PersonaLinkedMessage>(
  messages: readonly Message[],
  characterVersionIds: ReadonlySet<string>
): Set<string> {
  const messagesById = new Map(messages.map((message) => [message.id, message]));
  const result = new Set<string>();

  for (const message of messages) {
    if (message.role !== "assistant" && message.role !== "user") {
      continue;
    }

    let personaVersionId = message.persona_version_id;
    if (!personaVersionId && message.role === "user" && message.active_child_message_id) {
      personaVersionId =
        messagesById.get(message.active_child_message_id)?.persona_version_id ?? null;
    }

    if (personaVersionId && characterVersionIds.has(personaVersionId)) {
      result.add(message.id);
    }
  }

  return result;
}
