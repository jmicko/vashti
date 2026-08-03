import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { requestJson } from "./api";
import { attachmentReferences, isImageAttachment, uploadAttachmentToChat } from "./attachments";
import {
  activateMessagePath,
  activeMessageAttachments,
  activePathMessages,
  applyMessageVersionSelection,
  groupMessagesByParent,
  latestAssistantThinkingMode,
  latestAssistantModelValue,
  mergeStreamSegmentsByMessage,
  scrollMessageListToBottom,
  scrollMessageTopIntoListView,
  splitThinkingDelta,
  streamSegmentsFromRevision,
  streamingAssistantIdFromMessages,
  updateRevisionList,
  versionInfoForMessage
} from "./chatMessages";
import { ConfirmDialog, RetroLoader } from "./common";
import { StartChatComposer, type GenerationNotice } from "./Composer";
import { readGenerateEventStream } from "./generationStream";
import { MessageBubble, PendingOutgoingMessage } from "./MessageBubble";
import { ModelBackgroundLayer, modelBackgroundContainerStyle } from "./ModelBackground";
import { MessageTreeExplorer } from "./MessageTreeExplorer";
import { defaultToolPreferences, normalizeChatDetail } from "./toolPreferences";
import {
  deleteHostedPendingSend,
  getCachedHostedChat,
  getHostedPendingSend,
  saveCachedHostedChat,
  saveHostedPendingSend
} from "./privateChatStore";
import {
  modelParts,
  modelValue,
  personaModelValue,
  personaVersionIdFromValue,
  selectedModelBaseParts
} from "./modelSelection";
import type {
  AutoScrollMode,
  AttachmentInfo,
  AvailableTool,
  BranchScrollAnchor,
  ChatDetail,
  ChatInferenceSettings,
  ContextBlockSelection,
  ChatMessage,
  ChatResponse,
  ChatSyncResponse,
  ChatToolPreferences,
  ComposerAttachment,
  ComposerSubmitPayload,
  GenerateEvent,
  HostedPendingSend,
  ImageOpenHandler,
  ListMessagesResponse,
  MessageResponse,
  MessageStreamSegment,
  MessageVersion,
  BackendModelGroup,
  ModelInfo,
  Persona,
  PersonaVersion,
  ThinkingMode
} from "./types";

type HostedVersionMutation =
  | {
      key: string;
      kind: "root";
      chatId: string;
      messageId: string;
    }
  | {
      key: string;
      kind: "child";
      chatId: string;
      parentMessageId: string;
      messageId: string;
    }
  | {
      key: string;
      kind: "revision";
      chatId: string;
      messageId: string;
      revisionId: string;
    };

export function ChatView({
  chatId,
  error,
  queuedPrompt,
  selectedModel,
  selectedModelInfo,
  modelGroups,
  inferenceSettings,
  availableTools,
  personas,
  personaVersions,
  isTreeOpen,
  onTreeClose,
  onChatsChanged,
  onChatSettingsLoaded,
  onConversationSettingsSave,
  onImageOpen,
  onModelSelected,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  queuedPrompt: ({ chatId: string } & ComposerSubmitPayload) | null;
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  modelGroups: BackendModelGroup[];
  inferenceSettings: ChatInferenceSettings;
  availableTools: AvailableTool[];
  personas: Persona[];
  personaVersions: PersonaVersion[];
  isTreeOpen: boolean;
  onTreeClose: () => void;
  onChatsChanged: () => Promise<void>;
  onChatSettingsLoaded: (
    override: string | null | undefined,
    inferenceSettings?: ChatInferenceSettings,
    contextBlocks?: ContextBlockSelection[]
  ) => void;
  onConversationSettingsSave: () => Promise<void>;
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
  const [generationNotices, setGenerationNotices] = useState<GenerationNotice[]>([]);
  const [pendingPrompt, setPendingPrompt] = useState<ComposerSubmitPayload | null>(null);
  const [pendingSend, setPendingSend] = useState<HostedPendingSend | null>(null);
  const [isPendingSendLoading, setIsPendingSendLoading] = useState(true);
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>("auto");
  const [busyMessageId, setBusyMessageId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ChatMessage | null>(null);
  const [streamSegments, setStreamSegments] = useState<Record<string, MessageStreamSegment[]>>({});
  const messageListRef = useRef<HTMLElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRunRef = useRef(0);
  const pendingSendRef = useRef<HostedPendingSend | null>(null);
  const autoScrollModeRef = useRef<AutoScrollMode>("top");
  const userScrollIntentRef = useRef(false);
  const userScrollIntentTimeoutRef = useRef<number | null>(null);
  const scrollToBottomAfterLoadRef = useRef(false);
  const branchScrollAnchorRef = useRef<BranchScrollAnchor | null>(null);
  const generationJumpTargetRef = useRef<string | null>(null);
  const activeChatIdRef = useRef(chatId);
  const activeAssistantIdRef = useRef<string | null>(null);
  const chatStateRef = useRef<ChatDetail | null>(null);
  const messagesStateRef = useRef<ChatMessage[]>([]);
  const loadChatRef = useRef<((bypassCache?: boolean) => Promise<void>) | null>(null);
  const pendingVersionMutationsRef = useRef(new Map<string, HostedVersionMutation>());
  const isPersistingVersionMutationsRef = useRef(false);
  const pendingHostedCacheRef = useRef<{
    chat: ChatDetail;
    active_root_message_id: string | null;
    messages: ChatMessage[];
  } | null>(null);
  const isSavingHostedCacheRef = useRef(false);
  const thinkingStartedAtRef = useRef(new Map<string, number>());
  const [thinkingDurations, setThinkingDurations] = useState<Record<string, number>>({});
  const replaceChatState = useCallback((nextChat: ChatDetail | null) => {
    chatStateRef.current = nextChat;
    setChat(nextChat);
  }, []);
  const updateChatState = useCallback(
    (updater: (current: ChatDetail | null) => ChatDetail | null) => {
      replaceChatState(updater(chatStateRef.current));
    },
    [replaceChatState]
  );
  const replaceMessageList = useCallback((nextMessages: ChatMessage[]) => {
    messagesStateRef.current = nextMessages;
    setMessages(nextMessages);
  }, []);
  const updateMessageList = useCallback(
    (updater: (current: ChatMessage[]) => ChatMessage[]) => {
      replaceMessageList(updater(messagesStateRef.current));
    },
    [replaceMessageList]
  );
  const visibleMessages = useMemo(
    () => activePathMessages(messages, chat?.active_root_message_id ?? null),
    [chat?.active_root_message_id, messages]
  );
  const isActiveGenerationVisible =
    Boolean(activeAssistantId) &&
    visibleMessages.some((message) => message.id === activeAssistantId);
  const siblingGroups = useMemo(() => groupMessagesByParent(messages), [messages]);
  const chatContainsImages = useMemo(
    () => messages.some((message) => activeMessageAttachments(message).some(isImageAttachment)),
    [messages]
  );
  const modelImageWarning =
    selectedModelInfo && !selectedModelInfo.supports_images && chatContainsImages
      ? "This chat includes images. Images may not be supported by this model."
      : null;
  const currentGenerationNotice = generationNotices[generationNotices.length - 1] ?? null;
  activeChatIdRef.current = chatId;
  activeAssistantIdRef.current = activeAssistantId;
  chatStateRef.current = chat;
  messagesStateRef.current = messages;

  useEffect(() => {
    setGenerationNotices([]);
  }, [chatId]);

  useEffect(() => {
    const visibleMessageIds = new Set(visibleMessages.map((message) => message.id));
    setGenerationNotices((current) => {
      const next = current.filter((notice) => !visibleMessageIds.has(notice.messageId));
      return next.length === current.length ? current : next;
    });
  }, [visibleMessages]);

  const updatePendingSend = useCallback((next: HostedPendingSend | null) => {
    pendingSendRef.current = next;
    setPendingSend(next);
  }, []);

  const clearPendingSend = useCallback(
    async (pendingSendId: string) => {
      if (pendingSendRef.current?.id === pendingSendId) {
        updatePendingSend(null);
      }
      await deleteHostedPendingSend(pendingSendId).catch(() => undefined);
    },
    [updatePendingSend]
  );

  const failPendingSend = useCallback(
    (attempt: HostedPendingSend, message: string) => {
      const failedAttempt: HostedPendingSend = {
        ...attempt,
        status: "failed",
        error_text: message
      };
      updatePendingSend(failedAttempt);
      void saveHostedPendingSend(failedAttempt).catch(() => undefined);
    },
    [updatePendingSend]
  );

  const applyLoadedChat = useCallback(
    (nextChat: ChatDetail, activeRootMessageId: string | null, nextMessages: ChatMessage[]) => {
      const normalizedChat = normalizeChatDetail(nextChat);
      const normalizedMessages = nextMessages.map((message) => ({
        ...message,
        context_blocks: message.context_blocks ?? []
      }));
      replaceChatState({
        ...normalizedChat,
        active_root_message_id: activeRootMessageId
      });
      onChatSettingsLoaded(
        normalizedChat.system_prompt_override,
        normalizedChat.inference_settings,
        normalizedChat.context_blocks
      );
      thinkingStartedAtRef.current.clear();
      setThinkingDurations({});
      setStreamSegments({});
      replaceMessageList(normalizedMessages);
      const streamingAssistantId = streamingAssistantIdFromMessages(normalizedMessages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));
      const latestModel = latestAssistantModelValue(
        activePathMessages(normalizedMessages, activeRootMessageId)
      );
      onModelSelected(
        latestModel ??
          (normalizedChat.persona_version_id
            ? personaModelValue(normalizedChat.persona_version_id)
            : null) ??
          modelValue(normalizedChat.default_backend_id, normalizedChat.default_model_name)
      );
    },
    [onChatSettingsLoaded, onModelSelected, replaceChatState, replaceMessageList]
  );

  const queueHostedCacheSave = useCallback(
    (snapshot: {
      chat: ChatDetail;
      active_root_message_id: string | null;
      messages: ChatMessage[];
    }) => {
      pendingHostedCacheRef.current = snapshot;
      if (isSavingHostedCacheRef.current) {
        return;
      }

      isSavingHostedCacheRef.current = true;
      void (async () => {
        try {
          while (pendingHostedCacheRef.current) {
            const pending = pendingHostedCacheRef.current;
            pendingHostedCacheRef.current = null;
            await saveCachedHostedChat<ChatDetail, ChatMessage>(pending).catch(() => undefined);
          }
        } finally {
          isSavingHostedCacheRef.current = false;
          if (pendingHostedCacheRef.current) {
            queueHostedCacheSave(pendingHostedCacheRef.current);
          }
        }
      })();
    },
    []
  );

  const loadChat = useCallback(async (bypassCache = false) => {
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
      if (!bypassCache) {
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
      queueHostedCacheSave({
        chat: {
          ...syncResponse.chat,
          active_root_message_id: syncResponse.active_root_message_id
        },
        active_root_message_id: syncResponse.active_root_message_id,
        messages: nextMessages
      });
    } catch (chatError) {
      if (!displayedCachedChat) {
        setLoadError(chatError instanceof Error ? chatError.message : "Failed to load chat");
      }
    } finally {
      setIsLoading(false);
    }
  }, [applyLoadedChat, chatId, queueHostedCacheSave]);
  loadChatRef.current = loadChat;

  const refreshStreamingMessages = useCallback(async () => {
    try {
      const messageResponse = await requestJson<ListMessagesResponse>(
        `/api/chats/${chatId}/messages`
      );
      updateChatState((current) =>
        current
          ? { ...current, active_root_message_id: messageResponse.active_root_message_id }
          : current
      );
      replaceMessageList(messageResponse.messages);

      const streamingAssistantId = streamingAssistantIdFromMessages(messageResponse.messages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));

      if (!streamingAssistantId) {
        const chatResponse = await requestJson<ChatResponse>(`/api/chats/${chatId}`);
        const nextChat = {
          ...normalizeChatDetail(chatResponse.chat),
          active_root_message_id: messageResponse.active_root_message_id
        };
        replaceChatState(nextChat);
        onChatSettingsLoaded(
          nextChat.system_prompt_override,
          nextChat.inference_settings,
          nextChat.context_blocks
        );
        queueHostedCacheSave({
          chat: nextChat,
          active_root_message_id: messageResponse.active_root_message_id,
          messages: messageResponse.messages
        });
        await onChatsChanged();
      }
    } catch (refreshError) {
      setGenerationError(
        refreshError instanceof Error ? refreshError.message : "Failed to refresh generation"
      );
    }
  }, [
    chatId,
    onChatSettingsLoaded,
    onChatsChanged,
    queueHostedCacheSave,
    replaceChatState,
    replaceMessageList,
    updateChatState
  ]);

  useEffect(() => {
    if (
      !chat ||
      isLoading ||
      isGenerating ||
      isPersistingVersionMutationsRef.current ||
      pendingVersionMutationsRef.current.size > 0
    ) {
      return;
    }

    queueHostedCacheSave({
      chat,
      active_root_message_id: chat.active_root_message_id,
      messages
    });
  }, [chat, isGenerating, isLoading, messages, queueHostedCacheSave]);

  const streamAssistantResponse = useCallback(
    async (path: string, body: unknown, pendingAttempt?: HostedPendingSend) => {
      const runId = generationRunRef.current + 1;
      generationRunRef.current = runId;
      const controller = new AbortController();
      abortRef.current = controller;
      autoScrollModeRef.current = "top";
      setIsGenerating(true);
      setGenerationError(null);
      let acknowledged = false;
      let streamError: string | null = null;

      try {
        await readGenerateEventStream({
          path,
          body,
          signal: controller.signal,
          onEvent: (event) => {
            if (pendingAttempt && event.type === "message_start" && event.user_message) {
              acknowledged = true;
              void clearPendingSend(pendingAttempt.id);
            } else if (pendingAttempt && event.type === "error" && !acknowledged) {
              streamError = event.message;
            }
            applyGenerateEvent(event, runId);
          }
        });
        if (pendingAttempt && !acknowledged) {
          failPendingSend(
            pendingAttempt,
            streamError ?? "The server did not acknowledge the message."
          );
          setGenerationError(null);
          return;
        }
        await onChatsChanged();
      } catch (generateError) {
        if (generateError instanceof DOMException && generateError.name === "AbortError") {
          if (pendingAttempt && !acknowledged) {
            failPendingSend(pendingAttempt, "The send was interrupted before it was acknowledged.");
          }
          return;
        }

        const message =
          generateError instanceof Error ? generateError.message : "Generation failed";
        if (pendingAttempt && !acknowledged) {
          failPendingSend(pendingAttempt, message);
          setGenerationError(null);
        } else {
          setGenerationError(message);
          const assistantId = activeAssistantIdRef.current;
          if (assistantId) {
            queueGenerationNotice(assistantId, "error");
          }
        }
      } finally {
        if (generationRunRef.current === runId) {
          setIsGenerating(false);
          setActiveAssistantId(null);
          abortRef.current = null;
        }
      }
    },
    [clearPendingSend, failPendingSend, onChatsChanged]
  );

  const persistConversationSettingsForGeneration = useCallback(async () => {
    try {
      await onConversationSettingsSave();
      return null;
    } catch (settingsError) {
      const message =
        settingsError instanceof Error
          ? settingsError.message
          : "Failed to save conversation settings";
      setGenerationError(message);
      return message;
    }
  }, [onConversationSettingsSave]);

  const generate = useCallback(
    async (
      prompt: string,
      attachments: ComposerAttachment[] = [],
      thinkMode: ThinkingMode = "auto",
      promptInferenceSettings: ChatInferenceSettings = inferenceSettings
    ) => {
      if (isGenerating || pendingSendRef.current) {
        return;
      }

      const selected =
        selectedModelBaseParts([], personas, [], selectedModel, personaVersions) ??
        modelParts(selectedModel);
      const personaVersionId = personaVersionIdFromValue(selectedModel);
      const requestPath = `/api/chats/${chatId}/generate`;
      const requestBody: Record<string, unknown> = {
        user_message: { content_text: prompt },
        backend_id: selected?.backendId ?? null,
        model_name: selected?.modelName ?? null,
        persona_version_id: personaVersionId,
        think_mode: thinkModeToPayload(thinkMode),
        inference_settings: promptInferenceSettings,
        tool_preferences: chat?.tool_preferences ?? defaultToolPreferences,
        attachments: attachmentReferences(attachments)
      };
      const attempt: HostedPendingSend = {
        id: chatId,
        chat_id: chatId,
        prompt,
        attachments: persistedAttachmentMetadata(attachments),
        request_path: requestPath,
        request_body: requestBody,
        known_message_ids: messages.map((message) => message.id),
        status: "sending",
        error_text: null,
        created_at: Math.floor(Date.now() / 1000)
      };

      updatePendingSend(attempt);
      await saveHostedPendingSend(attempt).catch(() => undefined);

      const settingsError = await persistConversationSettingsForGeneration();
      if (settingsError) {
        failPendingSend(attempt, settingsError);
        setGenerationError(null);
        return;
      }

      await streamAssistantResponse(requestPath, requestBody, attempt);
    },
    [
      chat?.tool_preferences,
      inferenceSettings,
      chatId,
      isGenerating,
      messages,
      persistConversationSettingsForGeneration,
      personas,
      personaVersions,
      selectedModel,
      streamAssistantResponse,
      updatePendingSend,
      failPendingSend
    ]
  );

  useEffect(() => {
    void loadChat();
  }, [loadChat]);

  useEffect(() => {
    let cancelled = false;
    updatePendingSend(null);
    setIsPendingSendLoading(true);

    void getHostedPendingSend<HostedPendingSend>(chatId)
      .then((storedAttempt) => {
        if (cancelled || !storedAttempt || storedAttempt.chat_id !== chatId) {
          return;
        }

        const recoveredAttempt: HostedPendingSend =
          storedAttempt.status === "sending"
            ? {
                ...storedAttempt,
                status: "failed",
                error_text: "The page closed before the server acknowledged this message."
              }
            : storedAttempt;
        updatePendingSend(recoveredAttempt);
        if (recoveredAttempt !== storedAttempt) {
          void saveHostedPendingSend(recoveredAttempt).catch(() => undefined);
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) {
          setIsPendingSendLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [chatId, updatePendingSend]);

  useEffect(() => {
    if (isLoading || isPendingSendLoading || !pendingSend) {
      return;
    }
    if (acknowledgedMessageForAttempt(messages, pendingSend)) {
      void clearPendingSend(pendingSend.id);
    }
  }, [clearPendingSend, isLoading, isPendingSendLoading, messages, pendingSend]);

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
    if (
      !queuedPrompt ||
      isLoading ||
      isPendingSendLoading ||
      !chat ||
      isGenerating ||
      pendingSend
    ) {
      return;
    }

    onQueuedPromptConsumed();
    void generate(
      queuedPrompt.prompt,
      queuedPrompt.attachments,
      queuedPrompt.thinkMode,
      queuedPrompt.inferenceSettings !== undefined
        ? queuedPrompt.inferenceSettings
        : inferenceSettings
    );
  }, [
    chat,
    generate,
    isGenerating,
    isPendingSendLoading,
    isLoading,
    onQueuedPromptConsumed,
    queuedPrompt,
    pendingSend,
    inferenceSettings
  ]);

  useEffect(() => {
    if (
      !pendingPrompt ||
      isGenerating ||
      isLoading ||
      isPendingSendLoading ||
      pendingSend ||
      !chat
    ) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(
      prompt.prompt,
      prompt.attachments,
      prompt.thinkMode,
      prompt.inferenceSettings !== undefined ? prompt.inferenceSettings : inferenceSettings
    );
  }, [
    chat,
    generate,
    inferenceSettings,
    isGenerating,
    isLoading,
    isPendingSendLoading,
    pendingPrompt,
    pendingSend
  ]);

  useLayoutEffect(() => {
    if (isLoading) {
      return;
    }

    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const messageList = list;
    const finalMessageId = pendingSend
      ? `pending-${pendingSend.id}`
      : visibleMessages[visibleMessages.length - 1]?.id ?? null;
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
  }, [chatId, isLoading, pendingSend, visibleMessages]);

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
    const targetMessageId = generationJumpTargetRef.current;
    if (!targetMessageId || isLoading) {
      return;
    }

    const list = messageListRef.current;
    const messageElement = list?.querySelector<HTMLElement>(
      `[data-message-id="${targetMessageId}"]`
    );
    if (!list || !messageElement) {
      return;
    }

    scrollMessageTopIntoListView(list, messageElement);
    autoScrollModeRef.current = "top";
    generationJumpTargetRef.current = null;
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

  function queueGenerationNotice(
    messageId: string,
    kind: GenerationNotice["kind"]
  ) {
    const currentChat = chatStateRef.current;
    if (
      currentChat &&
      activePathMessages(
        messagesStateRef.current,
        currentChat.active_root_message_id
      ).some((message) => message.id === messageId)
    ) {
      return;
    }

    setGenerationNotices((current) => [
      ...current.filter((notice) => notice.messageId !== messageId),
      { messageId, kind }
    ]);
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
        setStreamSegments((current) => ({
          ...current,
          [event.assistant_message.id]: streamSegmentsFromRevision(
            event.assistant_message.active_revision?.content_text ?? "",
            event.assistant_message.active_revision?.thinking_text ?? ""
          )
        }));
        if (event.user_message && !event.user_message.parent_message_id) {
          updateChatState((current) =>
            current ? { ...current, active_root_message_id: event.user_message?.id ?? null } : current
          );
        }
        updateMessageList((current) => {
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
        queueGenerationNotice(event.assistant_message_id, "complete");
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
        updateChatState((current) =>
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
          queueGenerationNotice(event.assistant_message_id, "error");
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
    updateMessageList((current) =>
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
    updateMessageList((current) =>
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
    updateChatState((current) =>
      current ? { ...current, tool_preferences: nextPreferences } : current
    );

    try {
      const response = await requestJson<ChatResponse>(`/api/chats/${chatId}`, {
        method: "PATCH",
        body: JSON.stringify({ tool_preferences: nextPreferences })
      });
      replaceChatState(normalizeChatDetail(response.chat));
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
      setPendingPrompt({ prompt, attachments, thinkMode, inferenceSettings });
      await stopGeneration();
      return;
    }

    await generate(prompt, attachments, thinkMode, inferenceSettings);
  }

  async function retryPendingMessage() {
    const storedAttempt = pendingSendRef.current;
    if (!storedAttempt || isGenerating) {
      return;
    }

    const retryAttempt: HostedPendingSend = {
      ...storedAttempt,
      status: "sending",
      error_text: null
    };
    updatePendingSend(retryAttempt);
    await saveHostedPendingSend(retryAttempt).catch(() => undefined);

    try {
      const response = await requestJson<ListMessagesResponse>(
        `/api/chats/${chatId}/messages`
      );
      if (acknowledgedMessageForAttempt(response.messages, retryAttempt)) {
        updateChatState((current) =>
          current
            ? { ...current, active_root_message_id: response.active_root_message_id }
            : current
        );
        replaceMessageList(
          response.messages.map((message) => ({
            ...message,
            context_blocks: message.context_blocks ?? []
          }))
        );
        await clearPendingSend(retryAttempt.id);
        await onChatsChanged();
        return;
      }
    } catch {
      // The generation request below will surface the useful connection error.
    }

    const settingsError = await persistConversationSettingsForGeneration();
    if (settingsError) {
      failPendingSend(retryAttempt, settingsError);
      setGenerationError(null);
      return;
    }

    await streamAssistantResponse(
      retryAttempt.request_path,
      retryAttempt.request_body,
      retryAttempt
    );
  }

  function discardPendingMessage() {
    const storedAttempt = pendingSendRef.current;
    if (!storedAttempt || storedAttempt.status === "sending") {
      return;
    }
    void clearPendingSend(storedAttempt.id);
  }

  function replaceMessage(nextMessage: ChatMessage) {
    updateMessageList((current) =>
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
      selectedModelBaseParts([], personas, [], selectedModel, personaVersions) ??
      modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    if (await persistConversationSettingsForGeneration()) {
      return;
    }
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/branch`, {
      content_text: contentText,
      backend_id: selected?.backendId ?? null,
      model_name: selected?.modelName ?? null,
      persona_version_id: personaVersionId,
      think_mode: thinkModeToPayload(thinkingMode),
      inference_settings: inferenceSettings,
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
      selectedModelBaseParts([], personas, [], selectedModel, personaVersions) ??
      modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    if (await persistConversationSettingsForGeneration()) {
      return;
    }
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/regenerate`, {
      backend_id: selected?.backendId ?? message.backend_id,
      model_name: selected?.modelName ?? message.model_name,
      persona_version_id: personaVersionId,
      think_mode: thinkModeToPayload(thinkingMode),
      inference_settings: inferenceSettings,
      tool_preferences: chat?.tool_preferences ?? defaultToolPreferences,
      attachments: []
    });
  }

  async function continueMessage(message: ChatMessage) {
    if (
      isGenerating ||
      message.role !== "assistant" ||
      message.is_deleted ||
      !message.active_revision?.content_text.trim()
    ) {
      return;
    }

    if (await persistConversationSettingsForGeneration()) {
      return;
    }
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/continue`, {
      inference_settings: inferenceSettings,
      tool_preferences: chat?.tool_preferences ?? defaultToolPreferences
    });
  }

  function activatePathToMessage(
    messageId: string,
    missingMessage: string,
    targetRevisionId?: string
  ) {
    const currentChat = chatStateRef.current;
    if (!currentChat) {
      return false;
    }

    const selection = activateMessagePath({
      chat: currentChat,
      messages: messagesStateRef.current,
      targetMessageId: messageId,
      targetRevisionId
    });
    if (!selection.rootMessageId) {
      setGenerationError(missingMessage);
      return false;
    }

    setGenerationError(null);
    branchScrollAnchorRef.current = null;
    generationJumpTargetRef.current = messageId;
    replaceChatState(selection.chat);
    replaceMessageList(selection.messages);

    if (selection.chatChanged) {
      const mutation: HostedVersionMutation = {
        key: `root:${chatId}`,
        kind: "root",
        chatId,
        messageId: selection.rootMessageId
      };
      pendingVersionMutationsRef.current.set(mutation.key, mutation);
    }

    for (const revisionSelection of selection.revisionSelections) {
      const mutation: HostedVersionMutation = {
        key: `revision:${revisionSelection.messageId}`,
        kind: "revision",
        chatId,
        messageId: revisionSelection.messageId,
        revisionId: revisionSelection.revisionId
      };
      pendingVersionMutationsRef.current.set(mutation.key, mutation);
    }

    const changedParentIds = new Set(selection.changedMessages.map((message) => message.id));
    for (const childSelection of selection.childSelections) {
      if (!changedParentIds.has(childSelection.parentMessageId)) {
        continue;
      }
      const mutation: HostedVersionMutation = {
        key: `child:${childSelection.parentMessageId}`,
        kind: "child",
        chatId,
        parentMessageId: childSelection.parentMessageId,
        messageId: childSelection.messageId
      };
      pendingVersionMutationsRef.current.set(mutation.key, mutation);
    }

    void flushVersionMutations();
    return true;
  }

  function openTreeBranch(messageId: string, revisionId: string) {
    if (
      !activatePathToMessage(
        messageId,
        "That message version is no longer available in this chat.",
        revisionId
      )
    ) {
      return;
    }
    onTreeClose();
  }

  function jumpToActiveGeneration() {
    const assistantId = activeAssistantId;
    if (!assistantId || isActiveGenerationVisible) {
      return;
    }

    activatePathToMessage(
      assistantId,
      "The generating response is no longer available in this chat."
    );
  }

  function jumpToGenerationNotice() {
    const notice = generationNotices[generationNotices.length - 1];
    if (
      !notice ||
      !activatePathToMessage(
        notice.messageId,
        "The completed response is no longer available in this chat."
      )
    ) {
      return;
    }

    setGenerationNotices((current) =>
      current.filter((currentNotice) => currentNotice.messageId !== notice.messageId)
    );
  }

  function dismissGenerationNotice() {
    const notice = generationNotices[generationNotices.length - 1];
    if (!notice) {
      return;
    }

    setGenerationNotices((current) =>
      current.filter((currentNotice) => currentNotice.messageId !== notice.messageId)
    );
  }

  async function selectVersion(currentMessage: ChatMessage, version: MessageVersion) {
    const nextMessage = version.message;
    const nextRevision = version.revision;
    const isSameMessage = currentMessage.id === nextMessage.id;
    const isSameRevision = nextMessage.active_revision_id === nextRevision.id;
    if (isSameMessage && isSameRevision) {
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

    setGenerationError(null);

    const currentChat = chatStateRef.current;
    if (!currentChat) {
      return;
    }

    const selection = applyMessageVersionSelection({
      chat: currentChat,
      messages: messagesStateRef.current,
      currentMessage,
      nextMessage,
      nextRevision
    });
    replaceChatState(selection.chat);
    replaceMessageList(selection.messages);

    for (const revisionSelection of selection.revisionSelections) {
      const mutation: HostedVersionMutation = {
        key: `revision:${revisionSelection.messageId}`,
        kind: "revision",
        chatId,
        messageId: revisionSelection.messageId,
        revisionId: revisionSelection.revisionId
      };
      pendingVersionMutationsRef.current.set(mutation.key, mutation);
    }
    if (selection.messageChanged) {
      const mutation: HostedVersionMutation = currentMessage.parent_message_id
        ? {
            key: `child:${currentMessage.parent_message_id}`,
            kind: "child",
            chatId,
            parentMessageId: currentMessage.parent_message_id,
            messageId: nextMessage.id
          }
        : {
            key: `root:${chatId}`,
            kind: "root",
            chatId,
            messageId: nextMessage.id
      };
      pendingVersionMutationsRef.current.set(mutation.key, mutation);
    }

    void flushVersionMutations();
  }

  async function persistVersionMutation(mutation: HostedVersionMutation) {
    if (mutation.kind === "root") {
      await requestJson<ChatResponse>(`/api/chats/${mutation.chatId}/active-root`, {
        method: "PATCH",
        body: JSON.stringify({ active_root_message_id: mutation.messageId })
      });
      return;
    }

    if (mutation.kind === "child") {
      await requestJson<MessageResponse>(
        `/api/chats/${mutation.chatId}/messages/${mutation.parentMessageId}/active-child`,
        {
          method: "PATCH",
          body: JSON.stringify({ active_child_message_id: mutation.messageId })
        }
      );
      return;
    }

    await requestJson<MessageResponse>(
      `/api/chats/${mutation.chatId}/messages/${mutation.messageId}/active-revision`,
      {
        method: "PATCH",
        body: JSON.stringify({ active_revision_id: mutation.revisionId })
      }
    );
  }

  async function flushVersionMutations() {
    if (isPersistingVersionMutationsRef.current) {
      return;
    }

    isPersistingVersionMutationsRef.current = true;
    const failedMutations = new Map<
      string,
      { mutation: HostedVersionMutation; error: unknown }
    >();
    try {
      while (pendingVersionMutationsRef.current.size > 0) {
        const mutations = [...pendingVersionMutationsRef.current.values()].sort(
          (left, right) => versionMutationOrder(left) - versionMutationOrder(right)
        );
        pendingVersionMutationsRef.current.clear();
        for (const mutation of mutations) {
          try {
            await persistVersionMutation(mutation);
            failedMutations.delete(mutation.key);
          } catch (mutationError) {
            failedMutations.set(mutation.key, { mutation, error: mutationError });
          }
        }
      }

      const activeFailures = [...failedMutations.values()].filter(
        ({ mutation }) => mutation.chatId === activeChatIdRef.current
      );
      if (activeFailures.length > 0) {
        const lastError = activeFailures[activeFailures.length - 1].error;
        setGenerationError(
          lastError instanceof Error ? lastError.message : "Failed to switch version"
        );
        await loadChatRef.current?.(true);
      } else {
        const currentChat = chatStateRef.current;
        if (currentChat && currentChat.id === activeChatIdRef.current) {
          queueHostedCacheSave({
            chat: currentChat,
            active_root_message_id: currentChat.active_root_message_id,
            messages: messagesStateRef.current
          });
        }
      }
    } finally {
      isPersistingVersionMutationsRef.current = false;
      if (pendingVersionMutationsRef.current.size > 0) {
        void flushVersionMutations();
      }
    }
  }

  return (
    <div
      className={
        selectedModelInfo?.background_asset_id
          ? "chat-view chat-view-with-background"
          : "chat-view"
      }
      style={modelBackgroundContainerStyle(selectedModelInfo)}
    >
      <ModelBackgroundLayer background={selectedModelInfo} />
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
              visibleMessages.length > 0 || pendingSend
                ? "message-list message-list-buffered"
                : "message-list"
            }
            aria-label="Chat messages"
            ref={messageListRef}
            onScroll={handleMessageListScroll}
            onTouchMove={noteUserScrollIntent}
            onWheel={noteUserScrollIntent}
          >
            {visibleMessages.length === 0 && !pendingSend ? (
              <div className="message-list-empty">
                <p>Ready for messages.</p>
              </div>
            ) : (
              <>
                {visibleMessages.map((message) => (
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
                    streamSegmentsForMessage={(targetMessage) =>
                      streamSegments[targetMessage.id]
                    }
                    thinkingDurationSeconds={thinkingDurations[message.id] ?? null}
                    thinkingDurationForMessage={(targetMessage) =>
                      thinkingDurations[targetMessage.id] ?? null
                    }
                    onCopy={copyMessage}
                    onDelete={setDeleteTarget}
                    onBranch={branchMessage}
                    onEdit={editMessage}
                    onImageOpen={onImageOpen}
                    onRemoveAttachment={removeAttachment}
                    onUploadAttachment={uploadAttachment}
                    onRegenerate={regenerateMessage}
                    onContinue={continueMessage}
                    selectedModelInfo={selectedModelInfo}
                    modelAvatar={hostedModelAvatarForMessage(
                      message,
                      personaVersions,
                      modelGroups
                    )}
                    modelAvatarForMessage={(targetMessage) =>
                      hostedModelAvatarForMessage(
                        targetMessage,
                        personaVersions,
                        modelGroups
                      )
                    }
                  />
                ))}
                {pendingSend && (
                  <PendingOutgoingMessage
                    pendingSend={pendingSend}
                    onDiscard={discardPendingMessage}
                    onImageOpen={onImageOpen}
                    onRetry={() => void retryPendingMessage()}
                  />
                )}
              </>
            )}
          </section>
          <div className="chat-composer-wrap">
            <StartChatComposer
              isBusy={isGenerating || pendingSend?.status === "sending"}
              isDisabled={!selectedModel || isPendingSendLoading || Boolean(pendingSend)}
              isGenerating={isGenerating}
              placeholder={
                pendingSend
                  ? "Retry or discard the unsent message"
                  : selectedModel
                    ? "Message Vashti"
                    : "Select a model to continue"
              }
              selectedModelInfo={selectedModelInfo}
              availableTools={availableTools}
              toolPreferences={chat.tool_preferences}
              thinkingMode={thinkingMode}
              warning={modelImageWarning}
              onToolPreferencesChange={(nextPreferences) =>
                void updateChatToolPreferences(nextPreferences)
              }
              onThinkingModeChange={setThinkingMode}
              onJumpToGeneration={
                isGenerating && activeAssistantId && !isActiveGenerationVisible
                  ? jumpToActiveGeneration
                  : undefined
              }
              generationNotice={
                currentGenerationNotice
                  ? {
                      kind: currentGenerationNotice.kind,
                      count: generationNotices.length
                    }
                  : null
              }
              onJumpToGenerationNotice={
                currentGenerationNotice ? jumpToGenerationNotice : undefined
              }
              onDismissGenerationNotice={
                currentGenerationNotice ? dismissGenerationNotice : undefined
              }
              onStop={stopGeneration}
              onUploadAttachment={uploadAttachment}
              onRemoveAttachment={removeAttachment}
              onSubmit={submitPrompt}
            />
          </div>
        </>
      ) : null}
      {chat && isTreeOpen && (
        <MessageTreeExplorer
          activeRootMessageId={chat.active_root_message_id}
          messages={messages}
          onClose={onTreeClose}
          onOpenBranch={openTreeBranch}
        />
      )}
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

function versionMutationOrder(mutation: HostedVersionMutation) {
  switch (mutation.kind) {
    case "root":
      return 0;
    case "revision":
      return 1;
    case "child":
      return 2;
  }
}

function hostedModelAvatarForMessage(
  message: ChatMessage,
  versions: PersonaVersion[],
  modelGroups: BackendModelGroup[]
) {
  if (message.persona_version_id) {
    const version = versions.find((candidate) => candidate.id === message.persona_version_id);
    if (version) {
      return {
        displayName: version.display_name,
        assetId: version.avatar_asset_id,
        cropX: version.avatar_crop_x,
        cropY: version.avatar_crop_y,
        cropSize: version.avatar_crop_size
      };
    }
  }
  if (!message.backend_id || !message.model_name) {
    return null;
  }
  const model = modelGroups
    .find((group) => group.backend.id === message.backend_id)
    ?.models.find((candidate) => candidate.name === message.model_name);
  return model
    ? {
        displayName: model.name,
        assetId: model.avatar_asset_id,
        cropX: model.avatar_crop_x,
        cropY: model.avatar_crop_y,
        cropSize: model.avatar_crop_size
      }
    : null;
}

function thinkModeToPayload(mode: ThinkingMode) {
  return mode === "auto" ? null : mode;
}

function persistedAttachmentMetadata(attachments: ComposerAttachment[]): AttachmentInfo[] {
  return attachments
    .filter((attachment) => attachment.status === "uploaded")
    .map((attachment) => {
      const metadata = { ...attachment } as Partial<ComposerAttachment>;
      delete metadata.status;
      delete metadata.error;
      delete metadata.file;
      delete metadata.isExisting;
      return metadata as AttachmentInfo;
    });
}

function acknowledgedMessageForAttempt(
  messages: ChatMessage[],
  attempt: HostedPendingSend
): ChatMessage | null {
  const knownMessageIds = new Set(attempt.known_message_ids);
  const expectedAttachmentIds = attempt.attachments.map((attachment) => attachment.id).sort();

  return (
    messages.find((message) => {
      if (message.role !== "user" || knownMessageIds.has(message.id)) {
        return false;
      }
      if ((message.active_revision?.content_text ?? "").trim() !== attempt.prompt.trim()) {
        return false;
      }

      const messageAttachmentIds = activeMessageAttachments(message)
        .map((attachment) => attachment.id)
        .sort();
      return (
        messageAttachmentIds.length === expectedAttachmentIds.length &&
        messageAttachmentIds.every((id, index) => id === expectedAttachmentIds[index])
      );
    }) ?? null
  );
}
