import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Lock } from "lucide-react";
import { isImageAttachment, preparePrivateAttachment } from "./attachments";
import { privateStreamTestEnabled } from "./browserFlags";
import {
  activeMessageAttachments,
  activePathMessages,
  fallbackTitleFromPrompt,
  groupMessagesByParent,
  latestAssistantThinkingMode,
  latestAssistantModelValue,
  mergeStreamSegmentsByMessage,
  privateAttachmentsForMessage,
  privatePromptMessagesWithPersona,
  scrollMessageListToBottom,
  scrollMessageTopIntoListView,
  splitThinkingDelta,
  streamingAssistantIdFromMessages,
  syntheticStreamExpectedContent,
  syntheticStreamExpectedThinking,
  updateRevisionList,
  versionInfoForMessage
} from "./chatMessages";
import { ConfirmDialog, RetroLoader } from "./common";
import { StartChatComposer } from "./Composer";
import { readGenerateEventStream } from "./generationStream";
import { MessageBubble } from "./MessageBubble";
import {
  createPrivateMessage,
  getPrivateChat,
  listPrivateMessages,
  privateId,
  savePrivateChat,
  savePrivateMessage,
  savePrivateMessages,
  unixTimestamp,
  type PrivateChatDetail,
  type PrivateChatMessage,
  type PrivateChatMessageRevision,
  type PrivatePersona,
  type PrivatePersonaVersion
} from "./privateChatStore";
import {
  messageModelValue,
  modelParts,
  modelValue,
  personaVersionIdFromValue,
  privatePersonaForVersionId,
  privatePersonaModelValue,
  privatePersonaWithVersionForId,
  privatePersonaWithVersionForValue
} from "./modelSelection";
import type {
  AutoScrollMode,
  BranchScrollAnchor,
  ChatMessage,
  ChatInferenceSettings,
  ComposerAttachment,
  ComposerSubmitPayload,
  GenerateEvent,
  ImageOpenHandler,
  MessageStreamSegment,
  MessageVersion,
  ModelInfo,
  ThinkingMode
} from "./types";

export function PrivateChatView({
  chatId,
  error,
  queuedPrompt,
  selectedModel,
  selectedModelInfo,
  privatePersonas,
  privatePersonaVersions,
  systemPromptOverride,
  inferenceSettings,
  onChatSettingsLoaded,
  onImageOpen,
  onModelSelected,
  onPrivateChatsChanged,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  queuedPrompt: ({ chatId: string } & ComposerSubmitPayload) | null;
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  privatePersonas: PrivatePersona[];
  privatePersonaVersions: PrivatePersonaVersion[];
  systemPromptOverride: string | null;
  inferenceSettings: ChatInferenceSettings;
  onChatSettingsLoaded: (
    override: string | null | undefined,
    inferenceSettings?: ChatInferenceSettings
  ) => void;
  onImageOpen: ImageOpenHandler;
  onModelSelected: (value: string) => void;
  onPrivateChatsChanged: () => Promise<void>;
  onQueuedPromptConsumed: () => void;
}) {
  const [chat, setChat] = useState<PrivateChatDetail | null>(null);
  const [messages, setMessages] = useState<PrivateChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<ComposerSubmitPayload | null>(null);
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>("auto");
  const [busyMessageId, setBusyMessageId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<PrivateChatMessage | null>(null);
  const [streamSegments, setStreamSegments] = useState<Record<string, MessageStreamSegment[]>>({});
  const [streamTestStatus, setStreamTestStatus] = useState<string | null>(null);
  const [isPrivateBannerExpanded, setIsPrivateBannerExpanded] = useState(true);
  const messageListRef = useRef<HTMLElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRunRef = useRef(0);
  const autoScrollModeRef = useRef<AutoScrollMode>("top");
  const userScrollIntentRef = useRef(false);
  const userScrollIntentTimeoutRef = useRef<number | null>(null);
  const scrollToBottomAfterLoadRef = useRef(false);
  const branchScrollAnchorRef = useRef<BranchScrollAnchor | null>(null);
  const messagesRef = useRef<PrivateChatMessage[]>([]);
  const privateSaveChainRef = useRef<Promise<void>>(Promise.resolve());
  const privateBannerTimerRef = useRef<number | null>(null);
  const thinkingStartedAtRef = useRef(new Map<string, number>());
  const thinkingContentCursorRef = useRef(new Map<string, number>());
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
  const showStreamTest = privateStreamTestEnabled();
  const hasChat = Boolean(chat);

  const loadPrivateChat = useCallback(async () => {
    scrollToBottomAfterLoadRef.current = true;
    setIsLoading(true);
    setLoadError(null);

    try {
      const [nextChat, nextMessages] = await Promise.all([
        getPrivateChat(chatId),
        listPrivateMessages(chatId)
      ]);
      if (!nextChat) {
        throw new Error("Private chat not found on this device");
      }

      setChat(nextChat);
      onChatSettingsLoaded(nextChat.system_prompt_override, nextChat.inference_settings ?? {});
      thinkingStartedAtRef.current.clear();
      thinkingContentCursorRef.current.clear();
      setThinkingDurations({});
      setStreamSegments({});
      messagesRef.current = nextMessages;
      setMessages(nextMessages);
      const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));
      const latestModel = latestAssistantModelValue(
        activePathMessages(nextMessages, nextChat.active_root_message_id)
      );
      const defaultModel =
        nextChat.persona_version_id &&
        (privatePersonaForVersionId(privatePersonas, nextChat.persona_version_id) ||
          privatePersonaWithVersionForId(
            privatePersonas,
            privatePersonaVersions,
            nextChat.persona_version_id
          ))
          ? privatePersonaModelValue(nextChat.persona_version_id)
          : modelValue(nextChat.default_backend_id, nextChat.default_model_name);
      onModelSelected(latestModel ?? defaultModel);
    } catch (chatError) {
      setLoadError(
        chatError instanceof Error ? chatError.message : "Failed to load private chat"
      );
    } finally {
      setIsLoading(false);
    }
  }, [
    chatId,
    onChatSettingsLoaded,
    onModelSelected,
    privatePersonas,
    privatePersonaVersions
  ]);

  useEffect(() => {
    void loadPrivateChat();
  }, [loadPrivateChat]);

  useEffect(() => {
    setIsPrivateBannerExpanded(true);
  }, [chatId]);

  useEffect(() => {
    const latestModel = latestAssistantModelValue(visibleMessages);
    if (latestModel) {
      onModelSelected(latestModel);
    }
    setThinkingMode(latestAssistantThinkingMode(visibleMessages));
  }, [onModelSelected, visibleMessages]);

  useEffect(() => {
    if (privateBannerTimerRef.current) {
      window.clearTimeout(privateBannerTimerRef.current);
      privateBannerTimerRef.current = null;
    }

    if (!isPrivateBannerExpanded || !hasChat) {
      return;
    }

    privateBannerTimerRef.current = window.setTimeout(() => {
      setIsPrivateBannerExpanded(false);
      privateBannerTimerRef.current = null;
    }, 3000);

    return () => {
      if (privateBannerTimerRef.current) {
        window.clearTimeout(privateBannerTimerRef.current);
        privateBannerTimerRef.current = null;
      }
    };
  }, [chatId, hasChat, isPrivateBannerExpanded]);

  useEffect(() => {
    return () => {
      // Do not abort here. The stream continues writing into IndexedDB during app navigation.
      if (userScrollIntentTimeoutRef.current) {
        window.clearTimeout(userScrollIntentTimeoutRef.current);
      }
      if (privateBannerTimerRef.current) {
        window.clearTimeout(privateBannerTimerRef.current);
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
      void listPrivateMessages(chatId).then((nextMessages) => {
        messagesRef.current = nextMessages;
        setMessages(nextMessages);
        const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
        setActiveAssistantId(streamingAssistantId);
        setIsGenerating(Boolean(streamingAssistantId));
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [chatId, messages]);

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

  useEffect(() => {
    if (!queuedPrompt || isLoading || !chat || isGenerating) {
      return;
    }

    onQueuedPromptConsumed();
    void submitPrompt(
      queuedPrompt.prompt,
      queuedPrompt.attachments,
      undefined,
      queuedPrompt.thinkMode,
      queuedPrompt.systemPromptOverride !== undefined
        ? queuedPrompt.systemPromptOverride
        : systemPromptOverride,
      queuedPrompt.inferenceSettings !== undefined
        ? queuedPrompt.inferenceSettings
        : inferenceSettings
    );
  }, [
    chat,
    inferenceSettings,
    isGenerating,
    isLoading,
    onQueuedPromptConsumed,
    queuedPrompt,
    systemPromptOverride
  ]);

  useEffect(() => {
    if (!pendingPrompt || isGenerating || isLoading || !chat) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(
      prompt.prompt,
      prompt.attachments,
      prompt.thinkMode,
      prompt.systemPromptOverride !== undefined
        ? prompt.systemPromptOverride
        : systemPromptOverride,
      prompt.inferenceSettings !== undefined ? prompt.inferenceSettings : inferenceSettings
    );
  }, [chat, inferenceSettings, isGenerating, isLoading, pendingPrompt, systemPromptOverride]);

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

  function updateMessage(messageId: string, updater: (message: PrivateChatMessage) => PrivateChatMessage) {
    let changedMessage: PrivateChatMessage | null = null;
    const next = messagesRef.current.map((message) => {
      if (message.id !== messageId) {
        return message;
      }

      changedMessage = updater(message);
      return changedMessage;
    });

    messagesRef.current = next;
    if (changedMessage) {
      queuePrivateMessageSave(changedMessage);
    }
    setMessages(next);
  }

  function replacePrivateMessages(nextMessages: PrivateChatMessage[]) {
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
  }

  function queuePrivateMessageSave(message: PrivateChatMessage) {
    privateSaveChainRef.current = privateSaveChainRef.current
      .catch(() => undefined)
      .then(() => savePrivateMessage(message))
      .catch((saveError) => {
        setGenerationError(
          saveError instanceof Error ? saveError.message : "Failed to save private message"
        );
      });
  }

  async function streamPrivateAssistantResponse(
    assistantId: string,
    body: unknown,
    path = "/api/private/generate"
  ) {
    const runId = generationRunRef.current + 1;
    generationRunRef.current = runId;
    const controller = new AbortController();
    abortRef.current = controller;
    autoScrollModeRef.current = "top";
    setIsGenerating(true);
    setActiveAssistantId(assistantId);
    setGenerationError(null);
    setStreamSegments((current) => ({ ...current, [assistantId]: [] }));
    thinkingContentCursorRef.current.set(assistantId, 0);

    try {
      await readGenerateEventStream({
        path,
        body,
        signal: controller.signal,
        onEvent: (event) => applyPrivateGenerateEvent(event, runId)
      });
      await privateSaveChainRef.current;
      await onPrivateChatsChanged();
    } catch (generateError) {
      if (generateError instanceof DOMException && generateError.name === "AbortError") {
        return;
      }

      const message = generateError instanceof Error ? generateError.message : "Generation failed";
      setGenerationError(message);
      updateMessage(assistantId, (current) => ({
        ...current,
        status: "error",
        error_text: message,
        completed_at: unixTimestamp(),
        updated_at: unixTimestamp()
      }));
      clearStreamSegments(assistantId);
    } finally {
      if (generationRunRef.current === runId) {
        setIsGenerating(false);
        setActiveAssistantId(null);
        abortRef.current = null;
      }
    }
  }

  async function generate(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    thinkMode: ThinkingMode = "auto",
    promptSystemPromptOverride: string | null = systemPromptOverride,
    promptInferenceSettings: ChatInferenceSettings = inferenceSettings
  ) {
    if (!chat || isGenerating) {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona =
      privatePersonaWithVersionForValue(privatePersonas, privatePersonaVersions, selectedModel) ??
      (messagesRef.current.length === 0 && chat.persona_version_id
        ? privatePersonaWithVersionForId(
            privatePersonas,
            privatePersonaVersions,
            chat.persona_version_id
          )
        : null);
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    if (!selected) {
      setGenerationError("Select a model before sending a private message");
      return;
    }

    const now = unixTimestamp();
    const pathMessages = activePathMessages(messages, chat.active_root_message_id);
    const parent = pathMessages[pathMessages.length - 1] as PrivateChatMessage | undefined;
    const userMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: parent?.id ?? null,
      role: "user",
      contentText: prompt,
      createdAt: now
    });
    userMessage.attachments = privateAttachmentsForMessage(userMessage, attachments);
    const assistantMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: userMessage.id,
      role: "assistant",
      contentText: "",
      status: "streaming",
      backendId: selected.backendId,
      modelName: selected.modelName,
      personaId: selectedPrivatePersona?.id ?? null,
      personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
      personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
      thinkMode: thinkModeToPayload(thinkMode),
      createdAt: now
    });
    userMessage.active_child_message_id = assistantMessage.id;

    const nextMessages = [
      ...messages.map((message) =>
        parent && message.id === parent.id
          ? { ...message, active_child_message_id: userMessage.id, updated_at: now }
          : message
      ),
      userMessage,
      assistantMessage
    ];
    const title =
      chat.title === "Private Chat" && messages.length === 0
        ? fallbackTitleFromPrompt(prompt, "Private Chat")
        : chat.title;
    const nextChat = {
      ...chat,
      title,
      default_backend_id: selected.backendId,
      default_model_name: selected.modelName,
      persona_id: selectedPrivatePersona?.id ?? null,
      persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
      persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
      inference_settings: promptInferenceSettings,
      active_root_message_id: chat.active_root_message_id ?? userMessage.id,
      updated_at: now,
      last_message_at: now
    };

    setChat(nextChat);
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
    await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
    await onPrivateChatsChanged();

    await streamPrivateAssistantResponse(assistantMessage.id, {
      assistant_message_id: assistantMessage.id,
      backend_id: selected.backendId,
      model_name: selected.modelName,
      think_mode: thinkModeToPayload(thinkMode),
      inference_settings: promptInferenceSettings,
      messages: privatePromptMessagesWithPersona(
        nextMessages,
        nextChat.active_root_message_id,
        assistantMessage.id,
        selectedPrivatePersona,
        promptSystemPromptOverride
      ),
      attachments: []
    });
  }

  async function runPrivateStreamTest() {
    if (!chat || isGenerating) {
      return;
    }

    const contentTokens = 1600;
    const thinkingTokens = 220;
    const now = unixTimestamp();
    const pathMessages = activePathMessages(messagesRef.current, chat.active_root_message_id);
    const parent = pathMessages[pathMessages.length - 1] as PrivateChatMessage | undefined;
    const userMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: parent?.id ?? null,
      role: "user",
      contentText: `Synthetic private stream test: ${thinkingTokens} thinking chunks, ${contentTokens} content chunks.`,
      createdAt: now
    });
    const assistantMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: userMessage.id,
      role: "assistant",
      contentText: "",
      status: "streaming",
      modelName: "synthetic-stream-test",
      createdAt: now
    });
    userMessage.active_child_message_id = assistantMessage.id;

    const nextMessages = [
      ...messagesRef.current.map((message) =>
        parent && message.id === parent.id
          ? { ...message, active_child_message_id: userMessage.id, updated_at: now }
          : message
      ),
      userMessage,
      assistantMessage
    ];
    const nextChat = {
      ...chat,
      title: chat.title === "Private Chat" ? "Private Stream Test" : chat.title,
      active_root_message_id: chat.active_root_message_id ?? userMessage.id,
      updated_at: now,
      last_message_at: now
    };

    setStreamTestStatus("Running synthetic stream test...");
    setChat(nextChat);
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
    await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
    await onPrivateChatsChanged();

    await streamPrivateAssistantResponse(
      assistantMessage.id,
      {
        assistant_message_id: assistantMessage.id,
        content_tokens: contentTokens,
        thinking_tokens: thinkingTokens,
        delay_ms: 0
      },
      "/api/dev/private-stream-test"
    );
    await privateSaveChainRef.current;

    const saved = messagesRef.current.find((message) => message.id === assistantMessage.id);
    const content = saved?.active_revision?.content_text ?? "";
    const thinking = saved?.active_revision?.thinking_text ?? "";
    const expectedContent = syntheticStreamExpectedContent(contentTokens);
    const expectedThinking = syntheticStreamExpectedThinking(thinkingTokens);
    if (content === expectedContent && thinking === expectedThinking) {
      setStreamTestStatus(
        `Passed synthetic stream test: ${thinkingTokens + contentTokens} chunks preserved.`
      );
      return;
    }

    setStreamTestStatus(
      `Failed synthetic stream test: content ${content.length}/${expectedContent.length}, thinking ${thinking.length}/${expectedThinking.length}.`
    );
  }

  async function submitPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    _toolPreferences?: unknown,
    thinkMode: ThinkingMode = "auto",
    promptSystemPromptOverride: string | null = systemPromptOverride,
    promptInferenceSettings: ChatInferenceSettings = inferenceSettings
  ) {
    if (isGenerating) {
      setPendingPrompt({
        prompt,
        attachments,
        thinkMode,
        systemPromptOverride: promptSystemPromptOverride,
        inferenceSettings: promptInferenceSettings
      });
      await stopGeneration();
      return;
    }

    await generate(
      prompt,
      attachments,
      thinkMode,
      promptSystemPromptOverride,
      promptInferenceSettings
    );
  }

  async function stopGeneration() {
    const assistantId = activeAssistantId;
    generationRunRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setIsGenerating(false);
    setActiveAssistantId(null);

    if (assistantId) {
      updateMessage(assistantId, (current) => ({
        ...current,
        status: "stopped",
        done_reason: "stopped",
        completed_at: unixTimestamp(),
        updated_at: unixTimestamp()
      }));
      clearStreamSegments(assistantId);
    }
  }

  function applyPrivateGenerateEvent(event: GenerateEvent, runId: number) {
    if (generationRunRef.current !== runId) {
      return;
    }

    switch (event.type) {
      case "thinking_delta":
        {
          const orderedDelta = orderedPrivateThinkingDelta(
            event.assistant_message_id,
            event.delta
          );
          markThinkingStarted(event.assistant_message_id);
          appendStreamSegments(event.assistant_message_id, splitThinkingDelta(orderedDelta));
          appendMessageText(event.assistant_message_id, "thinking_text", orderedDelta);
        }
        break;
      case "content_delta":
        finishThinkingDuration(event.assistant_message_id);
        appendStreamSegments(event.assistant_message_id, [{ type: "content", text: event.delta }]);
        appendMessageText(event.assistant_message_id, "content_text", event.delta);
        break;
      case "message_done":
        finishThinkingDuration(event.assistant_message_id);
        updateMessage(event.assistant_message_id, (current) => ({
          ...current,
          status: "complete",
          done_reason: event.done_reason,
          stats: event.stats ?? null,
          completed_at: unixTimestamp(),
          updated_at: unixTimestamp()
        }));
        clearStreamSegments(event.assistant_message_id);
        setIsGenerating(false);
        setActiveAssistantId(null);
        abortRef.current = null;
        break;
      case "message_stopped":
        updateMessage(event.assistant_message_id, (current) => ({
          ...current,
          status: "stopped",
          done_reason: "stopped",
          completed_at: unixTimestamp(),
          updated_at: unixTimestamp()
        }));
        clearStreamSegments(event.assistant_message_id);
        break;
      case "error":
        setGenerationError(event.message);
        if (event.assistant_message_id) {
          updateMessage(event.assistant_message_id, (current) => ({
            ...current,
            status: "error",
            error_text: event.message,
            completed_at: unixTimestamp(),
            updated_at: unixTimestamp()
          }));
          clearStreamSegments(event.assistant_message_id);
        }
        break;
      case "message_start":
      case "chat_title":
        break;
    }
  }

  function appendMessageText(
    messageId: string,
    field: "content_text" | "thinking_text",
    delta: string
  ) {
    updateMessage(messageId, (message) => {
      const activeRevision = message.active_revision ?? {
        id: message.active_revision_id ?? privateId("private-revision"),
        content_text: "",
        thinking_text: "",
        source: "original",
        created_at: message.created_at
      };
      const nextRevision = {
        ...activeRevision,
        [field]: activeRevision[field] + delta
      };
      const revisions = updateRevisionList(message.revisions, nextRevision);

      return {
        ...message,
        status: "streaming",
        active_revision_id: nextRevision.id,
        active_revision: nextRevision,
        revisions,
        revision_count: revisions.length,
        updated_at: unixTimestamp()
      };
    });
  }

  function orderedPrivateThinkingDelta(messageId: string, delta: string) {
    const message = messagesRef.current.find((current) => current.id === messageId);
    const contentText = message?.active_revision?.content_text ?? "";
    const contentCursor = Array.from(contentText).length;
    const previousCursor = thinkingContentCursorRef.current.get(messageId) ?? 0;

    if (contentCursor === previousCursor) {
      return delta;
    }

    thinkingContentCursorRef.current.set(messageId, contentCursor);
    return `<VASHTI_CONTENT_CURSOR>${contentCursor}</VASHTI_CONTENT_CURSOR>${delta}`;
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
    try {
      const now = unixTimestamp();
      const revision: PrivateChatMessageRevision = {
        id: privateId("private-revision"),
        content_text: contentText,
        thinking_text: message.active_revision?.thinking_text ?? "",
        source: "edit",
        created_at: now
      };
      updateMessage(message.id, (current) => {
        const revisions = updateRevisionList(current.revisions, revision);
        const nextAttachments = privateAttachmentsForMessage(current, attachments, revision.id);
        return {
          ...current,
          active_revision_id: revision.id,
          active_revision: revision,
          revisions,
          revision_count: revisions.length,
          attachments: [...(current.attachments ?? []), ...nextAttachments],
          updated_at: now
        };
      });
      clearStreamSegments(message.id);
    } finally {
      setBusyMessageId(null);
    }
  }

  async function branchMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    if (!chat || isGenerating || message.role !== "user") {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona = privatePersonaWithVersionForValue(
      privatePersonas,
      privatePersonaVersions,
      selectedModel
    );
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    if (!selected) {
      setGenerationError("Select a model before sending a private branch");
      return;
    }

    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const now = unixTimestamp();
      const userMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: message.parent_message_id,
        role: "user",
        contentText,
        createdAt: now
      });
      userMessage.attachments = privateAttachmentsForMessage(userMessage, attachments);
      const assistantMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: userMessage.id,
        role: "assistant",
        contentText: "",
        status: "streaming",
        backendId: selected.backendId,
        modelName: selected.modelName,
        personaId: selectedPrivatePersona?.id ?? null,
        personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
        personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
        thinkMode: thinkModeToPayload(thinkingMode),
        createdAt: now
      });
      userMessage.active_child_message_id = assistantMessage.id;

      const nextMessages = [
        ...messagesRef.current.map((current) =>
          message.parent_message_id && current.id === message.parent_message_id
            ? { ...current, active_child_message_id: userMessage.id, updated_at: now }
            : current
        ),
        userMessage,
        assistantMessage
      ];
      const nextChat = {
        ...chat,
        default_backend_id: selected.backendId,
        default_model_name: selected.modelName,
        persona_id: selectedPrivatePersona?.id ?? null,
        persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
        persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
        inference_settings: inferenceSettings,
        active_root_message_id: message.parent_message_id
          ? chat.active_root_message_id
          : userMessage.id,
        updated_at: now,
        last_message_at: now
      };

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      await privateSaveChainRef.current;
      await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
      await onPrivateChatsChanged();

      await streamPrivateAssistantResponse(assistantMessage.id, {
        assistant_message_id: assistantMessage.id,
        backend_id: selected.backendId,
        model_name: selected.modelName,
        think_mode: thinkModeToPayload(thinkingMode),
        inference_settings: inferenceSettings,
        messages: privatePromptMessagesWithPersona(
          nextMessages,
          nextChat.active_root_message_id,
          assistantMessage.id,
          selectedPrivatePersona,
          systemPromptOverride
        ),
        attachments: []
      });
    } finally {
      setBusyMessageId(null);
    }
  }

  async function deleteMessage(message: ChatMessage) {
    setBusyMessageId(message.id);
    try {
      const now = unixTimestamp();
      updateMessage(message.id, (current) => ({
        ...current,
        is_deleted: true,
        status: "complete",
        active_revision: current.active_revision
          ? { ...current.active_revision, content_text: "", thinking_text: "" }
          : null,
        revisions: current.revisions.map((revision) => ({
          ...revision,
          content_text: "",
          thinking_text: ""
        })),
        updated_at: now
      }));
      setDeleteTarget(null);
    } finally {
      setBusyMessageId(null);
    }
  }

  async function regenerateMessage(message: ChatMessage) {
    if (!chat || isGenerating || message.role !== "assistant") {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona =
      privatePersonaWithVersionForValue(privatePersonas, privatePersonaVersions, selectedModel) ??
      privatePersonaWithVersionForId(
        privatePersonas,
        privatePersonaVersions,
        message.persona_version_id
      );
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    const backendId = selected?.backendId ?? message.backend_id;
    const modelName = selected?.modelName ?? message.model_name;
    if (!backendId || !modelName) {
      setGenerationError("Select a model before regenerating this private response");
      return;
    }

    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const now = unixTimestamp();
      const assistantMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: message.parent_message_id,
        role: "assistant",
        contentText: "",
        status: "streaming",
        backendId,
        modelName,
        personaId: selectedPrivatePersona?.id ?? null,
        personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
        personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
        thinkMode: thinkModeToPayload(thinkingMode),
        createdAt: now
      });

      const nextMessages = [
        ...messagesRef.current.map((current) =>
          message.parent_message_id && current.id === message.parent_message_id
            ? { ...current, active_child_message_id: assistantMessage.id, updated_at: now }
            : current
        ),
        assistantMessage
      ];
      const nextChat = {
        ...chat,
        default_backend_id: backendId,
        default_model_name: modelName,
        persona_id: selectedPrivatePersona?.id ?? null,
        persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
        persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
        inference_settings: inferenceSettings,
        active_root_message_id: message.parent_message_id
          ? chat.active_root_message_id
          : assistantMessage.id,
        updated_at: now,
        last_message_at: now
      };

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      await privateSaveChainRef.current;
      await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
      await onPrivateChatsChanged();

      await streamPrivateAssistantResponse(assistantMessage.id, {
        assistant_message_id: assistantMessage.id,
        backend_id: backendId,
        model_name: modelName,
        think_mode: thinkModeToPayload(thinkingMode),
        inference_settings: inferenceSettings,
        messages: privatePromptMessagesWithPersona(
          nextMessages,
          nextChat.active_root_message_id,
          assistantMessage.id,
          selectedPrivatePersona,
          systemPromptOverride
        ),
        attachments: []
      });
    } finally {
      setBusyMessageId(null);
    }
  }

  async function selectVersion(currentMessage: ChatMessage, version: MessageVersion) {
    const nextMessage = version.message as PrivateChatMessage;
    const nextRevision = version.revision as PrivateChatMessageRevision;
    const isSameMessage = currentMessage.id === nextMessage.id;
    const isSameRevision = nextMessage.active_revision_id === nextRevision.id;
    if ((isSameMessage && isSameRevision) || isGenerating || !chat) {
      return;
    }

    setBusyMessageId(currentMessage.id);
    setGenerationError(null);

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

    try {
      const now = unixTimestamp();
      const nextMessages = messagesRef.current.map((message) => {
        if (!isSameMessage) {
          if (currentMessage.parent_message_id && message.id === currentMessage.parent_message_id) {
            return { ...message, active_child_message_id: nextMessage.id, updated_at: now };
          }
        }

        if (message.id === nextMessage.id && !isSameRevision) {
          return {
            ...message,
            active_revision_id: nextRevision.id,
            active_revision: nextRevision,
            updated_at: now
          };
        }

        return message;
      });
      const nextChat =
        !isSameMessage && !currentMessage.parent_message_id
          ? { ...chat, active_root_message_id: nextMessage.id, updated_at: now }
          : chat;

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      const nextModelValue = messageModelValue(nextMessage);
      if (nextMessage.role === "assistant" && nextModelValue) {
        onModelSelected(nextModelValue);
      }

      await privateSaveChainRef.current;
      await Promise.all([
        nextChat === chat ? Promise.resolve() : savePrivateChat(nextChat),
        savePrivateMessages(nextMessages)
      ]);
      await onPrivateChatsChanged();
    } finally {
      setBusyMessageId(null);
    }
  }

  return (
    <div className="chat-view private-chat-view">
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
          <div
            className={
              isPrivateBannerExpanded
                ? "private-local-banner"
                : "private-local-banner private-local-banner-collapsed"
            }
          >
            <button
              type="button"
              className="private-local-toggle"
              aria-label={
                isPrivateBannerExpanded
                  ? "Collapse private chat notice"
                  : "Show private chat notice"
              }
              aria-expanded={isPrivateBannerExpanded}
              onClick={() => setIsPrivateBannerExpanded((expanded) => !expanded)}
            >
              <Lock />
            </button>
            {isPrivateBannerExpanded && (
              <>
                <span className="private-local-text">
                  Stored only in this browser. Clearing browser data can delete this chat.
                </span>
                {showStreamTest && (
                  <button
                    type="button"
                    className="stream-test-button"
                    disabled={isGenerating}
                    onClick={() => void runPrivateStreamTest()}
                  >
                    Run Stream Test
                  </button>
                )}
                {streamTestStatus && <small>{streamTestStatus}</small>}
              </>
            )}
          </div>
          <section
            className={
              visibleMessages.length > 0 ? "message-list message-list-buffered" : "message-list"
            }
            aria-label="Private chat messages"
            ref={messageListRef}
            onScroll={handleMessageListScroll}
            onTouchMove={noteUserScrollIntent}
            onWheel={noteUserScrollIntent}
          >
            {visibleMessages.length === 0 ? (
              <div className="message-list-empty">
                <p>Private chat ready.</p>
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
                  onDelete={(message) => setDeleteTarget(message as PrivateChatMessage)}
                  onBranch={branchMessage}
                  onEdit={editMessage}
                  onImageOpen={onImageOpen}
                  onUploadAttachment={preparePrivateAttachment}
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
              placeholder={selectedModel ? "Message private chat" : "Select a model to continue"}
              selectedModelInfo={selectedModelInfo}
              thinkingMode={thinkingMode}
              warning={modelImageWarning}
              onThinkingModeChange={setThinkingMode}
              onStop={stopGeneration}
              onUploadAttachment={preparePrivateAttachment}
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
          title="Delete Private Message"
          message="Delete this local message? Its text and thinking content will be scrubbed from this browser."
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
