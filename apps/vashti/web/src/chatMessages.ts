import { isImageAttachment } from "./attachments";
import { messageModelValue } from "./modelSelection";
import { privateId, unixTimestamp, type PrivateChatMessage, type PrivatePersona } from "./privateChatStore";
import type {
  AttachmentInfo,
  ChatMessage,
  ChatMessageRevision,
  ComposerAttachment,
  MessageStreamSegment,
  MessageVersion,
  ParsedThinkingText,
  ThinkingSegment,
  ThinkingMode,
  ToolUsageRecord,
  VersionInfo
} from "./types";

const rootSiblingGroupKey = "__root__";

export function parseThinkingText(rawThinkingText: string): ParsedThinkingText {
  const segments: ThinkingSegment[] = [];
  const markerPattern =
    /<VASHTI_(TOOL_USAGE|CONTENT_CURSOR)>([\s\S]*?)<\/VASHTI_(?:TOOL_USAGE|CONTENT_CURSOR)>/g;
  let visibleThinkingText = "";
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = markerPattern.exec(rawThinkingText)) !== null) {
    const textBefore = rawThinkingText.slice(cursor, match.index);
    pushThinkingTextSegment(segments, textBefore);
    visibleThinkingText += textBefore;

    if (match[1] === "TOOL_USAGE") {
      const usage = parseToolUsageRecord(match[2]);
      if (usage) {
        segments.push({ type: "tool", usage });
      }
      visibleThinkingText += "\n";
    }
    cursor = match.index + match[0].length;
  }

  const textAfter = rawThinkingText.slice(cursor);
  pushThinkingTextSegment(segments, textAfter);
  visibleThinkingText += textAfter;

  return {
    thinkingText: visibleThinkingText.replace(/\n{3,}/g, "\n\n").trim(),
    segments
  };
}

export function splitThinkingDelta(delta: string): MessageStreamSegment[] {
  const segments: MessageStreamSegment[] = [];
  const markerPattern =
    /<VASHTI_(TOOL_USAGE|CONTENT_CURSOR)>([\s\S]*?)<\/VASHTI_(?:TOOL_USAGE|CONTENT_CURSOR)>/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = markerPattern.exec(delta)) !== null) {
    const textBefore = delta.slice(cursor, match.index);
    if (textBefore) {
      segments.push({ type: "thinking", text: textBefore });
    }

    if (match[1] === "TOOL_USAGE") {
      const usage = parseToolUsageRecord(match[2]);
      if (usage) {
        segments.push({ type: "tool", usage });
      }
    }
    cursor = match.index + match[0].length;
  }

  const textAfter = delta.slice(cursor);
  if (textAfter) {
    segments.push({ type: "thinking", text: textAfter });
  }

  return segments;
}

export function messageDisplaySegmentsFromRevision(
  contentText: string,
  thinkingText: string
): MessageStreamSegment[] {
  if (!thinkingText.includes("<VASHTI_CONTENT_CURSOR>")) {
    return [];
  }

  const contentCodePoints = Array.from(contentText);
  const segments: MessageStreamSegment[] = [];
  const markerPattern =
    /<VASHTI_(TOOL_USAGE|CONTENT_CURSOR)>([\s\S]*?)<\/VASHTI_(?:TOOL_USAGE|CONTENT_CURSOR)>/g;
  let thinkingCursor = 0;
  let contentCursor = 0;
  let match: RegExpExecArray | null;

  while ((match = markerPattern.exec(thinkingText)) !== null) {
    const thinkingBefore = thinkingText.slice(thinkingCursor, match.index);
    if (thinkingBefore) {
      segments.push({ type: "thinking", text: thinkingBefore });
    }

    if (match[1] === "CONTENT_CURSOR") {
      const nextContentCursor = clampContentCursor(
        Number.parseInt(match[2], 10),
        contentCodePoints.length
      );
      appendContentSlice(segments, contentCodePoints, contentCursor, nextContentCursor);
      contentCursor = nextContentCursor;
    } else {
      const usage = parseToolUsageRecord(match[2]);
      if (usage) {
        segments.push({ type: "tool", usage });
      }
    }

    thinkingCursor = match.index + match[0].length;
  }

  const remainingThinking = thinkingText.slice(thinkingCursor);
  if (remainingThinking) {
    segments.push({ type: "thinking", text: remainingThinking });
  }
  appendContentSlice(segments, contentCodePoints, contentCursor, contentCodePoints.length);

  return mergeMessageStreamSegments([], segments);
}

function clampContentCursor(value: number, max: number) {
  if (!Number.isFinite(value)) {
    return max;
  }

  return Math.max(0, Math.min(max, value));
}

function appendContentSlice(
  segments: MessageStreamSegment[],
  contentCodePoints: string[],
  start: number,
  end: number
) {
  if (end <= start) {
    return;
  }

  segments.push({
    type: "content",
    text: contentCodePoints.slice(start, end).join("")
  });
}

export function mergeMessageStreamSegments(
  current: MessageStreamSegment[],
  incoming: MessageStreamSegment[]
) {
  const next = [...current];

  for (const segment of incoming) {
    const last = next[next.length - 1];
    if (last?.type === "thinking" && segment.type === "thinking") {
      next[next.length - 1] = { ...last, text: last.text + segment.text };
      continue;
    }
    if (last?.type === "content" && segment.type === "content") {
      next[next.length - 1] = { ...last, text: last.text + segment.text };
      continue;
    }

    next.push(segment);
  }

  return next;
}

export function mergeStreamSegmentsByMessage(
  current: Record<string, MessageStreamSegment[]>,
  messageId: string,
  segments: MessageStreamSegment[]
) {
  if (segments.length === 0) {
    return current;
  }

  return {
    ...current,
    [messageId]: mergeMessageStreamSegments(current[messageId] ?? [], segments)
  };
}

function pushThinkingTextSegment(segments: ThinkingSegment[], text: string) {
  const normalizedText = text.replace(/\n{3,}/g, "\n\n").trim();
  if (normalizedText) {
    segments.push({ type: "text", text: normalizedText });
  }
}

function parseToolUsageRecord(jsonText: string): ToolUsageRecord | null {
  try {
    const parsed = JSON.parse(jsonText) as Partial<ToolUsageRecord>;
    if (
      typeof parsed.name === "string" &&
      typeof parsed.summary === "string" &&
      typeof parsed.result === "string"
    ) {
      return {
        name: parsed.name,
        summary: parsed.summary,
        arguments: parsed.arguments ?? {},
        result: parsed.result
      };
    }
  } catch {
    // Keep malformed tool metadata out of the visible thinking text.
  }

  return null;
}

export function latestAssistantModelValue(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const value = messageModelValue(messages[index]);
    if (value) {
      return value;
    }
  }

  return null;
}

export function latestAssistantThinkingMode(messages: ChatMessage[]): ThinkingMode {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") {
      continue;
    }

    return thinkingModeFromMessage(message.think_mode);
  }

  return "auto";
}

function thinkingModeFromMessage(mode: string | null): ThinkingMode {
  switch (mode) {
    case "false":
    case "low":
    case "medium":
    case "high":
      return mode;
    default:
      return "auto";
  }
}

export function streamingAssistantIdFromMessages(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.status === "streaming") {
      return message.id;
    }
  }

  return null;
}

export function privatePromptMessages(
  messages: ChatMessage[],
  activeRootMessageId: string | null,
  stopBeforeMessageId: string
) {
  return activePathMessages(messages, activeRootMessageId)
    .filter((message) => message.id !== stopBeforeMessageId)
    .filter((message) => !message.is_deleted)
    .map((message) => {
      const attachmentPayload = privateAttachmentPromptPayload(activeMessageAttachments(message));
      return {
        role: message.role,
        content_text: withPrivateAttachmentText(
          message.active_revision?.content_text ?? "",
          attachmentPayload.text
        ),
        thinking_text: message.active_revision?.thinking_text || null,
        images: attachmentPayload.images
      };
    })
    .filter((message) => message.content_text.trim() !== "" || message.images.length > 0);
}

export function privatePromptMessagesWithPersona(
  messages: ChatMessage[],
  activeRootMessageId: string | null,
  stopBeforeMessageId: string,
  persona: PrivatePersona | null,
  systemPromptOverride?: string | null
) {
  const promptMessages = privatePromptMessages(messages, activeRootMessageId, stopBeforeMessageId);
  const systemPrompt =
    systemPromptOverride === undefined || systemPromptOverride === null
      ? persona?.current_version.system_prompt.trim()
      : systemPromptOverride.trim();
  if (!systemPrompt) {
    return promptMessages;
  }

  return [
    {
      role: "system",
      content_text: systemPrompt,
      thinking_text: null,
      images: []
    },
    ...promptMessages
  ];
}

function privateAttachmentPromptPayload(attachments: AttachmentInfo[]) {
  const textParts: string[] = [];
  const images: string[] = [];

  for (const attachment of attachments) {
    if (isImageAttachment(attachment)) {
      const imageBase64 = imageBase64FromDataUrl(attachment.data_url);
      if (imageBase64) {
        images.push(imageBase64);
      }
      continue;
    }

    if (attachment.text_content) {
      textParts.push(
        `Attachment: ${attachment.original_filename}\n\n${attachment.text_content}`
      );
    }
  }

  return {
    text: textParts.join("\n\n---\n\n"),
    images
  };
}

function withPrivateAttachmentText(content: string, attachmentText: string) {
  if (!attachmentText) {
    return content;
  }

  return `${content.trim()}\n\n${attachmentText}`.trim();
}

function imageBase64FromDataUrl(dataUrl: string | undefined) {
  const commaIndex = dataUrl?.indexOf(",") ?? -1;
  return commaIndex >= 0 ? dataUrl?.slice(commaIndex + 1) ?? "" : "";
}

export function activeMessageAttachments(message: ChatMessage) {
  const attachments = message.attachments ?? [];
  if (!message.active_revision_id) {
    return attachments;
  }

  return attachments.filter(
    (attachment) =>
      !attachment.revision_id || attachment.revision_id === message.active_revision_id
  );
}

export function composerAttachmentFromExisting(attachment: AttachmentInfo): ComposerAttachment {
  return {
    ...attachment,
    status: "uploaded",
    isExisting: true
  };
}

export function privateAttachmentsForMessage(
  message: PrivateChatMessage,
  attachments: ComposerAttachment[],
  revisionId = message.active_revision_id
) {
  return attachments
    .filter((attachment) => attachment.status === "ready" || attachment.status === "uploaded")
    .map((attachment) =>
      privateAttachmentForMessage(
        message,
        attachment,
        attachment.isExisting ? privateId("private-attachment") : attachment.id,
        revisionId
      )
    );
}

function privateAttachmentForMessage(
  message: PrivateChatMessage,
  attachment: AttachmentInfo,
  id: string,
  revisionId: string | null
) {
  return {
    id,
    chat_id: message.chat_id,
    message_id: message.id,
    revision_id: revisionId,
    original_filename: attachment.original_filename,
    mime_type: attachment.mime_type,
    size_bytes: attachment.size_bytes,
    attachment_kind: attachment.attachment_kind,
    created_at: attachment.created_at ?? unixTimestamp(),
    data_url: attachment.data_url,
    text_content: attachment.text_content
  };
}

export function fallbackTitleFromPrompt(prompt: string, fallback: string) {
  const title = prompt
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 5)
    .join(" ")
    .trim()
    .replace(/^["'`*#.,:;\s]+|["'`*#.,:;\s]+$/g, "");

  return title || fallback;
}

export function syntheticStreamExpectedThinking(count: number) {
  return Array.from({ length: count }, (_, index) => `think-${String(index + 1).padStart(5, "0")} `).join("");
}

export function syntheticStreamExpectedContent(count: number) {
  return Array.from({ length: count }, (_, index) => {
    const token = index + 1;
    return token % 17 === 0
      ? `\nchunk-${String(token).padStart(5, "0")};`
      : `tok-${String(token).padStart(5, "0")} `;
  }).join("");
}

export function revisionsForMessage(message: ChatMessage) {
  const revisions = message.revisions.length
    ? message.revisions
    : message.active_revision
      ? [message.active_revision]
      : [];

  return [...revisions].sort(compareRevisionsByCreatedAt);
}

export function updateRevisionList(
  revisions: ChatMessageRevision[],
  nextRevision: ChatMessageRevision
) {
  const nextRevisions = revisions.length ? [...revisions] : [];
  const revisionIndex = nextRevisions.findIndex((revision) => revision.id === nextRevision.id);
  if (revisionIndex >= 0) {
    nextRevisions[revisionIndex] = nextRevision;
  } else {
    nextRevisions.push(nextRevision);
  }

  return nextRevisions.sort(compareRevisionsByCreatedAt);
}

export function activePathMessages(messages: ChatMessage[], activeRootMessageId: string | null) {
  const messagesById = new Map(messages.map((message) => [message.id, message]));
  const rootMessages = messages
    .filter((message) => !message.parent_message_id)
    .sort(compareMessagesByCreatedAt);
  const path: ChatMessage[] = [];
  const seen = new Set<string>();
  let currentId: string | null = activeRootMessageId ?? rootMessages[0]?.id ?? null;

  while (currentId && !seen.has(currentId)) {
    const message = messagesById.get(currentId);
    if (!message) {
      break;
    }

    path.push(message);
    seen.add(currentId);
    currentId = message.active_child_message_id;
  }

  return path;
}

export function groupMessagesByParent(messages: ChatMessage[]) {
  const groups = new Map<string, ChatMessage[]>();

  for (const message of messages) {
    const key = parentGroupKey(message.parent_message_id);
    const siblings = groups.get(key) ?? [];
    siblings.push(message);
    groups.set(key, siblings);
  }

  for (const siblings of groups.values()) {
    siblings.sort(compareMessagesByCreatedAt);
  }

  return groups;
}

export function parentGroupKey(parentMessageId: string | null) {
  return parentMessageId ?? rootSiblingGroupKey;
}

function compareMessagesByCreatedAt(left: ChatMessage, right: ChatMessage) {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

function compareRevisionsByCreatedAt(left: ChatMessageRevision, right: ChatMessageRevision) {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

export function compareVersionsByCreatedAt(left: MessageVersion, right: MessageVersion) {
  return (
    left.revision.created_at - right.revision.created_at ||
    left.message.created_at - right.message.created_at ||
    left.revision.id.localeCompare(right.revision.id)
  );
}

export function versionsForMessage(
  message: ChatMessage,
  siblingGroups: Map<string, ChatMessage[]>
) {
  const siblings = siblingGroups.get(parentGroupKey(message.parent_message_id)) ?? [];
  return siblings
    .flatMap((sibling) =>
      revisionsForMessage(sibling).map((revision) => ({
        message: sibling,
        revision
      }))
    )
    .sort(compareVersionsByCreatedAt);
}

export function versionInfoForMessage(
  message: ChatMessage,
  siblingGroups: Map<string, ChatMessage[]>,
  selectVersion: (message: ChatMessage, version: MessageVersion) => void
): VersionInfo | null {
  const versions = versionsForMessage(message, siblingGroups);
  if (versions.length < 2 || !message.active_revision_id) {
    return null;
  }

  const index = versions.findIndex(
    (version) =>
      version.message.id === message.id && version.revision.id === message.active_revision_id
  );
  if (index < 0) {
    return null;
  }

  const previousVersion = versions[index - 1] ?? null;
  const nextVersion = versions[index + 1] ?? null;

  return {
    index,
    total: versions.length,
    canPrevious: Boolean(previousVersion),
    canNext: Boolean(nextVersion),
    onPrevious: () => {
      if (previousVersion) {
        selectVersion(message, previousVersion);
      }
    },
    onNext: () => {
      if (nextVersion) {
        selectVersion(message, nextVersion);
      }
    }
  };
}

export function scrollMessageTopIntoListView(list: HTMLElement, messageElement: HTMLElement) {
  const styles = window.getComputedStyle(list);
  const topPadding = Number.parseFloat(styles.paddingTop) || 0;
  const topOffset =
    messageElement.getBoundingClientRect().top - list.getBoundingClientRect().top - topPadding;
  list.scrollTop += topOffset;
}

export function scrollMessageListToBottom(list: HTMLElement) {
  list.scrollTop = Math.max(0, list.scrollHeight - list.clientHeight);
}
