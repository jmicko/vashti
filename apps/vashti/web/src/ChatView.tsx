import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { requestJson } from "./api";
import { attachmentReferences, isImageAttachment, uploadAttachmentToChat } from "./attachments";
import {
  activeMessageAttachments,
  activePathMessages,
  groupMessagesByParent,
  latestAssistantThinkingMode,
  latestAssistantModelValue,
  mergeStreamSegmentsByMessage,
  scrollMessageListToBottom,
  scrollMessageTopIntoListView,
  splitThinkingDelta,
  streamingAssistantIdFromMessages,
  updateRevisionList,
  versionInfoForMessage
} from "./chatMessages";
import { ConfirmDialog, RetroLoader } from "./common";
import { StartChatComposer } from "./Composer";
import { readGenerateEventStream } from "./generationStream";
import { MessageBubble } from "./MessageBubble";
import { defaultToolPreferences, normalizeChatDetail } from "./toolPreferences";
import { getCachedHostedChat, saveCachedHostedChat } from "./privateChatStore";
import {
  modelParts,
  modelValue,
  personaModelValue,
  personaVersionIdFromValue,
  selectedModelBaseParts
} from "./modelSelection";
import type {
  AutoScrollMode,
  AvailableTool,
  BranchScrollAnchor,
  ChatDetail,
  ChatMessage,
  ChatResponse,
  ChatSyncResponse,
  ChatToolPreferences,
  ComposerAttachment,
  ComposerSubmitPayload,
  GenerateEvent,
  ImageOpenHandler,
  ListMessagesResponse,
  MessageResponse,
  MessageStreamSegment,
  MessageVersion,
  ModelInfo,
  Persona,
  ThinkingMode
} from "./types";

export function ChatView({
  chatId,
  error,
  queuedPrompt,
  selectedModel,
  selectedModelInfo,
  availableTools,
  personas,
  onChatsChanged,
  onImageOpen,
  onModelSelected,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  queuedPrompt: ({ chatId: string } & ComposerSubmitPayload) | null;
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  availableTools: AvailableTool[];
  personas: Persona[];
  onChatsChanged: () => Promise<void>;
  onImageOpen: ImageOpenHandler;
  onModelSelected: (value: string) => void;
  onQueuedPromptConsumed: () => void;
}) {
  const [chat, setChat] = useState<ChatDetail | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<ComposerSubmitPayload | null>(null);
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>("auto");
  const [busyMessageId, setBusyMessageId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ChatMessage | null>(null);
  const [streamSegments, setStreamSegments] = useState<Record<string, MessageStreamSegment[]>>({});
  const messageListRef = useRef<HTMLElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRunRef = useRef(0);
  const autoScrollModeRef = useRef<AutoScrollMode>("top");
  const userScrollIntentRef = useRef(false);
  const userScrollIntentTimeoutRef = useRef<number | null>(null);
  const scrollToBottomAfterLoadRef = useRef(false);
  const branchScrollAnchorRef = useRef<BranchScrollAnchor | null>(null);
  const thinkingStartedAtRef = useRef(new Map<string, number>());
  const [thinkingDurations, setThinkingDurations] = useState<Record<string, number>>({});
  const visibleMessages = useMemo(
    () => activePathMessages(messages, chat?.active_root_message_id ?? null),
    [chat?.active_root_message_id, messages]
  );
  const siblingGroups = useMemo(() => groupMessagesByParent(messages), [messages]);
  const chatContainsImages = useMemo(
    () => messages.some((message) => activeMessageAttachments(message).some(isImageAttachment)),
    [messages]
  );
  const modelImageWarning =
    selectedModelInfo && !selectedModelInfo.supports_images && chatContainsImages
      ? "This chat includes images. Images may not be supported by this model."
      : null;

  const applyLoadedChat = useCallback(
    (nextChat: ChatDetail, activeRootMessageId: string | null, nextMessages: ChatMessage[]) => {
      const normalizedChat = normalizeChatDetail(nextChat);
      setChat({
        ...normalizedChat,
        active_root_message_id: activeRootMessageId
      });
      thinkingStartedAtRef.current.clear();
      setThinkingDurations({});
      setStreamSegments({});
      setMessages(nextMessages);
      const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));
      const latestModel = latestAssistantModelValue(
        activePathMessages(nextMessages, activeRootMessageId)
      );
      onModelSelected(
        latestModel ??
          (normalizedChat.persona_version_id
            ? personaModelValue(normalizedChat.persona_version_id)
            : null) ??
          modelValue(normalizedChat.default_backend_id, normalizedChat.default_model_name)
      );
    },
    [onModelSelected]
  );

  const loadChat = useCallback(async () => {
    scrollToBottomAfterLoadRef.current = true;
    setIsLoading(true);
    setLoadError(null);
    let cachedChat: {
      chat: ChatDetail;
      active_root_message_id: string | null;
      messages: ChatMessage[];
      updated_at: number;
    } | null = null;
    let displayedCachedChat = false;

    try {
      cachedChat = await getCachedHostedChat<ChatDetail, ChatMessage>(chatId);
      if (cachedChat) {
        applyLoadedChat(
          cachedChat.chat,
          cachedChat.active_root_message_id,
          cachedChat.messages
        );
        displayedCachedChat = true;
        setIsLoading(false);
      }
    } catch {
      // Hosted chat cache is best-effort. The server remains authoritative.
    }

    try {
      const syncUrl =
        cachedChat?.updated_at === undefined
          ? `/api/chats/${chatId}/sync`
          : `/api/chats/${chatId}/sync?known_updated_at=${cachedChat.updated_at}`;
      const syncResponse = await requestJson<ChatSyncResponse>(syncUrl);
      if (!syncResponse.changed && cachedChat) {
        applyLoadedChat(
          syncResponse.chat,
          syncResponse.active_root_message_id,
          cachedChat.messages
        );
        return;
      }

      const nextMessages = syncResponse.messages;
      if (!nextMessages) {
        throw new Error("Chat sync did not include messages");
      }

      scrollToBottomAfterLoadRef.current = true;
      applyLoadedChat(
        syncResponse.chat,
        syncResponse.active_root_message_id,
        nextMessages
      );
      await saveCachedHostedChat<ChatDetail, ChatMessage>({
        chat: {
          ...syncResponse.chat,
          active_root_message_id: syncResponse.active_root_message_id
        },
        active_root_message_id: syncResponse.active_root_message_id,
        messages: nextMessages
      }).catch(() => undefined);
    } catch (chatError) {
      if (!displayedCachedChat) {
        setLoadError(chatError instanceof Error ? chatError.message : "Failed to load chat");
      }
    } finally {
      setIsLoading(false);
    }
  }, [applyLoadedChat, chatId]);

  const refreshStreamingMessages = useCallback(async () => {
    try {
      const messageResponse = await requestJson<ListMessagesResponse>(
        `/api/chats/${chatId}/messages`
      );
      setChat((current) =>
        current
          ? { ...current, active_root_message_id: messageResponse.active_root_message_id }
          : current
      );
      setMessages(messageResponse.messages);

      const streamingAssistantId = streamingAssistantIdFromMessages(messageResponse.messages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));

      if (!streamingAssistantId) {
        const chatResponse = await requestJson<ChatResponse>(`/api/chats/${chatId}`);
        const nextChat = {
          ...normalizeChatDetail(chatResponse.chat),
          active_root_message_id: messageResponse.active_root_message_id
        };
        setChat(nextChat);
        await saveCachedHostedChat<ChatDetail, ChatMessage>({
          chat: nextChat,
          active_root_message_id: messageResponse.active_root_message_id,
          messages: messageResponse.messages
        }).catch(() => undefined);
        await onChatsChanged();
      }
    } catch (refreshError) {
      setGenerationError(
        refreshError instanceof Error ? refreshError.message : "Failed to refresh generation"
      );
    }
  }, [chatId, onChatsChanged]);

  useEffect(() => {
    if (!chat || isLoading || isGenerating) {
      return;
    }

    void saveCachedHostedChat<ChatDetail, ChatMessage>({
      chat,
      active_root_message_id: chat.active_root_message_id,
      messages
    }).catch(() => undefined);
  }, [chat, isGenerating, isLoading, messages]);

  const streamAssistantResponse = useCallback(
    async (path: string, body: unknown) => {
      const runId = generationRunRef.current + 1;
      generationRunRef.current = runId;
      const controller = new AbortController();
      abortRef.current = controller;
      autoScrollModeRef.current = "top";
      setIsGenerating(true);
      setGenerationError(null);

      try {
        await readGenerateEventStream({
          path,
          body,
          signal: controller.signal,
          onEvent: (event) => applyGenerateEvent(event, runId)
        });
        await onChatsChanged();
      } catch (generateError) {
        if (generateError instanceof DOMException && generateError.name === "AbortError") {
          return;
        }

        setGenerationError(
          generateError instanceof Error ? generateError.message : "Generation failed"
        );
      } finally {
        if (generationRunRef.current === runId) {
          setIsGenerating(false);
          setActiveAssistantId(null);
          abortRef.current = null;
        }
      }
    },
    [onChatsChanged]
  );

  const generate = useCallback(
    async (
      prompt: string,
      attachments: ComposerAttachment[] = [],
      thinkMode: ThinkingMode = "auto"
    ) => {
      if (isGenerating) {
        return;
      }

      const selected =
        selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
      const personaVersionId = personaVersionIdFromValue(selectedModel);
      await streamAssistantResponse(`/api/chats/${chatId}/generate`, {
        user_message: { content_text: prompt },
        backend_id: selected?.backendId ?? null,
        model_name: selected?.modelName ?? null,
        persona_version_id: personaVersionId,
        think_mode: thinkModeToPayload(thinkMode),
        tool_preferences: chat?.tool_preferences ?? defaultToolPreferences,
        attachments: attachmentReferences(attachments)
      });
    },
    [chat?.tool_preferences, chatId, isGenerating, personas, selectedModel, streamAssistantResponse]
  );

  useEffect(() => {
    void loadChat();
  }, [loadChat]);

  useEffect(() => {
    const latestModel = latestAssistantModelValue(visibleMessages);
    if (latestModel) {
      onModelSelected(latestModel);
    }
    setThinkingMode(latestAssistantThinkingMode(visibleMessages));
  }, [onModelSelected, visibleMessages]);

  useEffect(() => {
    return () => {
      if (userScrollIntentTimeoutRef.current) {
        window.clearTimeout(userScrollIntentTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const hasStreamingMessage = messages.some(
      (message) => message.role === "assistant" && message.status === "streaming"
    );
    if (!hasStreamingMessage || abortRef.current) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshStreamingMessages();
    }, 1000);

    return () => window.clearInterval(interval);
  }, [messages, refreshStreamingMessages]);

  useEffect(() => {
    if (!queuedPrompt || isLoading || !chat || isGenerating) {
      return;
    }

    onQueuedPromptConsumed();
    void generate(queuedPrompt.prompt, queuedPrompt.attachments, queuedPrompt.thinkMode);
  }, [
    chat,
    generate,
    isGenerating,
    isLoading,
    onQueuedPromptConsumed,
    queuedPrompt
  ]);

  useEffect(() => {
    if (!pendingPrompt || isGenerating || isLoading || !chat) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(prompt.prompt, prompt.attachments, prompt.thinkMode);
  }, [chat, generate, isGenerating, isLoading, pendingPrompt]);

  useLayoutEffect(() => {
    if (isLoading) {
      return;
    }

    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const messageList = list;
    const finalMessageId = visibleMessages[visibleMessages.length - 1]?.id ?? null;
    const finalMessage = finalMessageId
      ? messageList.querySelector<HTMLElement>(`[data-message-id="${finalMessageId}"]`)
      : null;

    function updateMessageListBuffer() {
      if (!finalMessage) {
        messageList.style.setProperty("--message-list-buffer-height", "0px");
        return;
      }

      const styles = window.getComputedStyle(messageList);
      const topPadding = Number.parseFloat(styles.paddingTop) || 0;
      const bottomPadding = Number.parseFloat(styles.paddingBottom) || 0;
      const rowGap = Number.parseFloat(styles.rowGap || styles.gap) || 0;
      const availableSpace =
        messageList.clientHeight -
        finalMessage.getBoundingClientRect().height -
        topPadding -
        bottomPadding -
        rowGap;
      const bufferHeight = Math.max(0, Math.floor(availableSpace));
      messageList.style.setProperty("--message-list-buffer-height", `${bufferHeight}px`);
    }

    updateMessageListBuffer();

    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(updateMessageListBuffer);
    resizeObserver?.observe(messageList);
    if (finalMessage) {
      resizeObserver?.observe(finalMessage);
    }
    window.addEventListener("resize", updateMessageListBuffer);
    window.visualViewport?.addEventListener("resize", updateMessageListBuffer);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updateMessageListBuffer);
      window.visualViewport?.removeEventListener("resize", updateMessageListBuffer);
    };
  }, [chatId, isLoading, visibleMessages]);

  useLayoutEffect(() => {
    if (isLoading || !scrollToBottomAfterLoadRef.current) {
      return;
    }

    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const messageList = list;
    const lastMessage = visibleMessages[visibleMessages.length - 1];
    const lastMessageElement = lastMessage
      ? messageList.querySelector<HTMLElement>(`[data-message-id="${lastMessage.id}"]`)
      : null;
    function scrollToLatestMessage() {
      if (lastMessageElement) {
        scrollMessageTopIntoListView(messageList, lastMessageElement);
      } else {
        messageList.scrollTop = 0;
      }
    }

    scrollToLatestMessage();
    const frame = window.requestAnimationFrame(scrollToLatestMessage);
    const settleTimeout = window.setTimeout(scrollToLatestMessage, 120);
    const lateSettleTimeout = window.setTimeout(scrollToLatestMessage, 420);
    scrollToBottomAfterLoadRef.current = false;

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settleTimeout);
      window.clearTimeout(lateSettleTimeout);
    };
  }, [chatId, isLoading, visibleMessages]);

  useLayoutEffect(() => {
    const anchor = branchScrollAnchorRef.current;
    if (!anchor || isLoading) {
      return;
    }

    const list = messageListRef.current;
    const messageElement = list?.querySelector<HTMLElement>(
      `[data-message-id="${anchor.messageId}"]`
    );
    if (!list || !messageElement) {
      branchScrollAnchorRef.current = null;
      return;
    }

    const anchorElement =
      messageElement.querySelector<HTMLElement>(".version-switcher") ?? messageElement;
    const nextTopOffset = anchorElement.getBoundingClientRect().top - list.getBoundingClientRect().top;
    list.scrollTop += nextTopOffset - anchor.topOffset;
    branchScrollAnchorRef.current = null;
  }, [isLoading, visibleMessages]);

  useLayoutEffect(() => {
    if (!isGenerating || !activeAssistantId || autoScrollModeRef.current === "paused") {
      return;
    }

    const list = messageListRef.current;
    const activeMessage = list?.querySelector<HTMLElement>(
      `[data-message-id="${activeAssistantId}"]`
    );
    if (!list || !activeMessage) {
      return;
    }

    const listRect = list.getBoundingClientRect();
    const messageRect = activeMessage.getBoundingClientRect();
    const topOffset = messageRect.top - listRect.top;
    const bottomOverflow = messageRect.bottom - listRect.bottom;

    if (autoScrollModeRef.current === "bottom") {
      scrollMessageListToBottom(list);
      return;
    }

    if (bottomOverflow <= 0 || topOffset <= 8) {
      return;
    }

    list.scrollTop += Math.min(bottomOverflow + 16, topOffset - 8);
  }, [activeAssistantId, isGenerating, visibleMessages]);

  function noteUserScrollIntent() {
    if (!isGenerating) {
      return;
    }

    userScrollIntentRef.current = true;
    if (userScrollIntentTimeoutRef.current) {
      window.clearTimeout(userScrollIntentTimeoutRef.current);
    }
    userScrollIntentTimeoutRef.current = window.setTimeout(() => {
      userScrollIntentRef.current = false;
    }, 240);

    window.requestAnimationFrame(updateAutoScrollModeFromScrollPosition);
  }

  function handleMessageListScroll() {
    if (!isGenerating || !userScrollIntentRef.current) {
      return;
    }

    updateAutoScrollModeFromScrollPosition();
  }

  function updateAutoScrollModeFromScrollPosition() {
    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const activeMessage = activeAssistantId
      ? list.querySelector<HTMLElement>(`[data-message-id="${activeAssistantId}"]`)
      : null;
    if (!activeMessage) {
      const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight <= 12;
      autoScrollModeRef.current = atBottom ? "bottom" : "paused";
      return;
    }

    const listRect = list.getBoundingClientRect();
    const messageRect = activeMessage.getBoundingClientRect();
    const bottomOverflow = messageRect.bottom - listRect.bottom;
    if (bottomOverflow > 12) {
      autoScrollModeRef.current = "paused";
      return;
    }

    const topOffset = messageRect.top - listRect.top;
    autoScrollModeRef.current = topOffset <= 8 ? "bottom" : "top";
  }

  function clearThinkingDuration(messageId: string) {
    thinkingStartedAtRef.current.delete(messageId);
    setThinkingDurations((current) => {
      if (!(messageId in current)) {
        return current;
      }

      const next = { ...current };
      delete next[messageId];
      return next;
    });
  }

  function markThinkingStarted(messageId: string) {
    if (!thinkingStartedAtRef.current.has(messageId)) {
      thinkingStartedAtRef.current.set(messageId, Date.now());
    }
  }

  function finishThinkingDuration(messageId: string) {
    const startedAt = thinkingStartedAtRef.current.get(messageId);
    if (!startedAt) {
      return;
    }

    thinkingStartedAtRef.current.delete(messageId);
    const durationSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    setThinkingDurations((current) =>
      current[messageId] ? current : { ...current, [messageId]: durationSeconds }
    );
  }

  function applyGenerateEvent(event: GenerateEvent, runId: number) {
    if (generationRunRef.current !== runId && event.type !== "chat_title") {
      return;
    }

    switch (event.type) {
      case "message_start":
        autoScrollModeRef.current = "top";
        setActiveAssistantId(event.assistant_message.id);
        clearThinkingDuration(event.assistant_message.id);
        setStreamSegments((current) => ({ ...current, [event.assistant_message.id]: [] }));
        if (event.user_message && !event.user_message.parent_message_id) {
          setChat((current) =>
            current ? { ...current, active_root_message_id: event.user_message?.id ?? null } : current
          );
        }
        setMessages((current) => {
          const parentUpdates = new Map<string, string>();
          if (event.user_message?.parent_message_id) {
            parentUpdates.set(event.user_message.parent_message_id, event.user_message.id);
          }
          if (event.assistant_message.parent_message_id) {
            parentUpdates.set(
              event.assistant_message.parent_message_id,
              event.assistant_message.id
            );
          }

          const incomingIds = new Set(
            [event.user_message?.id, event.assistant_message.id].filter(
              (id): id is string => Boolean(id)
            )
          );
          const now = Math.floor(Date.now() / 1000);
          const existingMessages = current
            .filter((message) => !incomingIds.has(message.id))
            .map((message) => {
              const activeChildId = parentUpdates.get(message.id);
              return activeChildId
                ? { ...message, active_child_message_id: activeChildId, updated_at: now }
                : message;
            });

          return [
            ...existingMessages,
            ...(event.user_message ? [event.user_message] : []),
            event.assistant_message
          ];
        });
        break;
      case "thinking_delta":
        markThinkingStarted(event.assistant_message_id);
        appendStreamSegments(event.assistant_message_id, splitThinkingDelta(event.delta));
        appendMessageText(event.assistant_message_id, "thinking_text", event.delta);
        break;
      case "content_delta":
        finishThinkingDuration(event.assistant_message_id);
        appendStreamSegments(event.assistant_message_id, [{ type: "content", text: event.delta }]);
        appendMessageText(event.assistant_message_id, "content_text", event.delta);
        break;
      case "message_done":
        finishThinkingDuration(event.assistant_message_id);
        updateMessageStatus(
          event.assistant_message_id,
          "complete",
          event.done_reason,
          event.stats ?? null
        );
        clearStreamSegments(event.assistant_message_id);
        if (generationRunRef.current === runId) {
          setIsGenerating(false);
          setActiveAssistantId(null);
          abortRef.current = null;
        }
        break;
      case "chat_title":
        setChat((current) =>
          current && current.id === event.chat_id ? { ...current, title: event.title } : current
        );
        break;
      case "message_stopped":
        finishThinkingDuration(event.assistant_message_id);
        updateMessageStatus(event.assistant_message_id, "stopped", "stopped");
        clearStreamSegments(event.assistant_message_id);
        break;
      case "error":
        setGenerationError(event.message);
        if (event.assistant_message_id) {
          finishThinkingDuration(event.assistant_message_id);
          updateMessageStatus(event.assistant_message_id, "error", "error");
          clearStreamSegments(event.assistant_message_id);
        }
        break;
    }
  }

  function appendMessageText(
    messageId: string,
    field: "content_text" | "thinking_text",
    delta: string
  ) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) {
          return message;
        }

        const activeRevision = message.active_revision ?? {
          id: message.active_revision_id ?? `${message.id}-active`,
          content_text: "",
          thinking_text: "",
          source: "original",
          created_at: message.created_at
        };

        return {
          ...message,
          status: "streaming",
          active_revision: {
            ...activeRevision,
            [field]: activeRevision[field] + delta
          },
          revisions: updateRevisionList(message.revisions, {
            ...activeRevision,
            [field]: activeRevision[field] + delta
          })
        };
      })
    );
  }

  function appendStreamSegments(messageId: string, segments: MessageStreamSegment[]) {
    setStreamSegments((current) => mergeStreamSegmentsByMessage(current, messageId, segments));
  }

  function clearStreamSegments(messageId: string) {
    setStreamSegments((current) => {
      if (!current[messageId]) {
        return current;
      }
      const next = { ...current };
      delete next[messageId];
      return next;
    });
  }

  function updateMessageStatus(
    messageId: string,
    status: string,
    doneReason: string | null,
    stats?: ChatMessage["stats"]
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              status,
              done_reason: doneReason,
              stats: stats === undefined ? message.stats : stats,
              completed_at: Math.floor(Date.now() / 1000)
            }
          : message
      )
    );
  }

  async function stopGeneration() {
    const assistantId = activeAssistantId;
    generationRunRef.current += 1;
    if (!assistantId) {
      abortRef.current?.abort();
      setIsGenerating(false);
      setActiveAssistantId(null);
      abortRef.current = null;
      return;
    }

    try {
      await requestJson(`/api/chats/${chatId}/messages/${assistantId}/stop`, { method: "POST" });
      updateMessageStatus(assistantId, "stopped", "stopped");
    } catch (stopError) {
      setGenerationError(stopError instanceof Error ? stopError.message : "Failed to stop");
    } finally {
      abortRef.current?.abort();
      setIsGenerating(false);
      setActiveAssistantId(null);
    }
  }

  async function updateChatToolPreferences(nextPreferences: ChatToolPreferences) {
    setChat((current) =>
      current ? { ...current, tool_preferences: nextPreferences } : current
    );

    try {
      const response = await requestJson<ChatResponse>(`/api/chats/${chatId}`, {
        method: "PATCH",
        body: JSON.stringify({ tool_preferences: nextPreferences })
      });
      setChat(normalizeChatDetail(response.chat));
    } catch (updateError) {
      setGenerationError(
        updateError instanceof Error ? updateError.message : "Failed to update tool settings"
      );
      void loadChat();
    }
  }

  async function uploadAttachment(file: File) {
    return { ...(await uploadAttachmentToChat(chatId, file)), status: "uploaded" as const };
  }

  async function removeAttachment(attachment: ComposerAttachment) {
    if (
      attachment.isExisting ||
      attachment.status !== "uploaded" ||
      attachment.id.startsWith("pending-")
    ) {
      return;
    }

    await requestJson(`/api/attachments/${attachment.id}`, { method: "DELETE" });
  }

  async function submitPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    _toolPreferences?: ChatToolPreferences,
    thinkMode: ThinkingMode = "auto"
  ) {
    if (isGenerating) {
      setPendingPrompt({ prompt, attachments, thinkMode });
      await stopGeneration();
      return;
    }

    await generate(prompt, attachments, thinkMode);
  }

  function replaceMessage(nextMessage: ChatMessage) {
    setMessages((current) =>
      current.map((message) => (message.id === nextMessage.id ? nextMessage : message))
    );
  }

  async function copyMessage(message: ChatMessage) {
    const content = message.active_revision?.content_text ?? "";
    if (!content || message.is_deleted) {
      return;
    }

    await navigator.clipboard.writeText(content);
    setCopiedMessageId(message.id);
    window.setTimeout(() => {
      setCopiedMessageId((current) => (current === message.id ? null : current));
    }, 1200);
  }

  async function editMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const response = await requestJson<MessageResponse>(
        `/api/chats/${chatId}/messages/${message.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            content_text: contentText,
            attachments: attachmentReferences(attachments)
          })
        }
      );
      replaceMessage(response.message);
      clearStreamSegments(message.id);
      await onChatsChanged();
    } catch (editError) {
      setGenerationError(editError instanceof Error ? editError.message : "Failed to edit message");
      throw editError;
    } finally {
      setBusyMessageId(null);
    }
  }

  async function branchMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    if (isGenerating || message.role !== "user") {
      return;
    }

    const selected =
      selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/branch`, {
      content_text: contentText,
      backend_id: selected?.backendId ?? null,
      model_name: selected?.modelName ?? null,
      persona_version_id: personaVersionId,
      think_mode: thinkModeToPayload(thinkingMode),
      tool_preferences: chat?.tool_preferences ?? defaultToolPreferences,
      attachments: attachmentReferences(attachments)
    });
  }

  async function deleteMessage(message: ChatMessage) {
    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const response = await requestJson<MessageResponse>(
        `/api/chats/${chatId}/messages/${message.id}`,
        { method: "DELETE" }
      );
      replaceMessage(response.message);
      setDeleteTarget(null);
      await onChatsChanged();
    } catch (deleteError) {
      setGenerationError(
        deleteError instanceof Error ? deleteError.message : "Failed to delete message"
      );
    } finally {
      setBusyMessageId(null);
    }
  }

  async function regenerateMessage(message: ChatMessage) {
    if (isGenerating || message.role !== "assistant") {
      return;
    }

    const selected =
      selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/regenerate`, {
      backend_id: selected?.backendId ?? message.backend_id,
      model_name: selected?.modelName ?? message.model_name,
      persona_version_id: personaVersionId,
      think_mode: thinkModeToPayload(thinkingMode),
      tool_preferences: chat?.tool_preferences ?? defaultToolPreferences,
      attachments: []
    });
  }

  async function selectVersion(currentMessage: ChatMessage, version: MessageVersion) {
    const nextMessage = version.message;
    const nextRevision = version.revision;
    const isSameMessage = currentMessage.id === nextMessage.id;
    const isSameRevision = nextMessage.active_revision_id === nextRevision.id;
    if ((isSameMessage && isSameRevision) || isGenerating) {
      return;
    }

    const list = messageListRef.current;
    const messageElement = list?.querySelector<HTMLElement>(
      `[data-message-id="${currentMessage.id}"]`
    );
    const anchorElement =
      messageElement?.querySelector<HTMLElement>(".version-switcher") ?? messageElement;
    if (list && anchorElement) {
      branchScrollAnchorRef.current = {
        messageId: nextMessage.id,
        topOffset: anchorElement.getBoundingClientRect().top - list.getBoundingClientRect().top
      };
    }

    setBusyMessageId(currentMessage.id);
    setGenerationError(null);

    try {
      if (!isSameMessage) {
        if (currentMessage.parent_message_id) {
          const response = await requestJson<MessageResponse>(
            `/api/chats/${chatId}/messages/${currentMessage.parent_message_id}/active-child`,
            {
              method: "PATCH",
              body: JSON.stringify({ active_child_message_id: nextMessage.id })
            }
          );
          replaceMessage(response.message);
        } else {
          const response = await requestJson<ChatResponse>(`/api/chats/${chatId}/active-root`, {
            method: "PATCH",
            body: JSON.stringify({ active_root_message_id: nextMessage.id })
          });
          setChat(response.chat);
        }

        if (nextMessage.role === "assistant") {
          if (nextMessage.persona_version_id) {
            onModelSelected(personaModelValue(nextMessage.persona_version_id));
          } else if (nextMessage.backend_id && nextMessage.model_name) {
            onModelSelected(modelValue(nextMessage.backend_id, nextMessage.model_name));
          }
        }
      }

      if (!isSameRevision) {
        const response = await requestJson<MessageResponse>(
          `/api/chats/${chatId}/messages/${nextMessage.id}/active-revision`,
          {
            method: "PATCH",
            body: JSON.stringify({ active_revision_id: nextRevision.id })
          }
        );
        replaceMessage(response.message);
      }
    } catch (selectError) {
      setGenerationError(
        selectError instanceof Error ? selectError.message : "Failed to switch version"
      );
    } finally {
      setBusyMessageId(null);
    }
  }

  return (
    <div className="chat-view">
      {isLoading ? (
        <div className="empty-state">
          <RetroLoader />
        </div>
      ) : loadError ? (
        <div className="empty-state">
          <p className="error">{loadError}</p>
        </div>
      ) : chat ? (
        <>
          <section
            className={
              visibleMessages.length > 0 ? "message-list message-list-buffered" : "message-list"
            }
            aria-label="Chat messages"
            ref={messageListRef}
            onScroll={handleMessageListScroll}
            onTouchMove={noteUserScrollIntent}
            onWheel={noteUserScrollIntent}
          >
            {visibleMessages.length === 0 ? (
              <div className="message-list-empty">
                <p>Ready for messages.</p>
              </div>
            ) : (
              visibleMessages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  versionInfo={versionInfoForMessage(
                    message,
                    siblingGroups,
                    (targetMessage, version) => {
                      void selectVersion(targetMessage, version);
                    }
                  )}
                  copied={copiedMessageId === message.id}
                  isBusy={busyMessageId === message.id}
                  isGenerating={isGenerating}
                  streamSegments={streamSegments[message.id]}
                  thinkingDurationSeconds={thinkingDurations[message.id] ?? null}
                  onCopy={copyMessage}
                  onDelete={setDeleteTarget}
                  onBranch={branchMessage}
                  onEdit={editMessage}
                  onImageOpen={onImageOpen}
                  onRemoveAttachment={removeAttachment}
                  onUploadAttachment={uploadAttachment}
                  onRegenerate={regenerateMessage}
                  selectedModelInfo={selectedModelInfo}
                />
              ))
            )}
          </section>
          <div className="chat-composer-wrap">
            <StartChatComposer
              isBusy={isGenerating}
              isDisabled={!selectedModel}
              isGenerating={isGenerating}
              placeholder={selectedModel ? "Message Vashti" : "Select a model to continue"}
              selectedModelInfo={selectedModelInfo}
              availableTools={availableTools}
              toolPreferences={chat.tool_preferences}
              thinkingMode={thinkingMode}
              warning={modelImageWarning}
              onToolPreferencesChange={(nextPreferences) =>
                void updateChatToolPreferences(nextPreferences)
              }
              onThinkingModeChange={setThinkingMode}
              onStop={stopGeneration}
              onUploadAttachment={uploadAttachment}
              onRemoveAttachment={removeAttachment}
              onSubmit={submitPrompt}
            />
          </div>
        </>
      ) : null}
      {(error || generationError) && (
        <p className="error chat-view-error">{generationError ?? error}</p>
      )}
      {deleteTarget && (
        <ConfirmDialog
          title="Delete Message"
          message="Delete this message? Its text and thinking content will be scrubbed, but the chat path will remain intact."
          confirmLabel="Delete"
          isBusy={busyMessageId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void deleteMessage(deleteTarget)}
        />
      )}
    </div>
  );
}

function thinkModeToPayload(mode: ThinkingMode) {
  return mode === "auto" ? null : mode;
}
