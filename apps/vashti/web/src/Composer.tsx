import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { Brain, Paperclip, SendHorizontal, Square, Wrench, X } from "lucide-react";
import {
  AttachmentChipList,
  attachmentAcceptTypes,
  isImageAttachment,
  newPendingAttachmentId,
  pendingComposerAttachment
} from "./attachments";
import { RetroLoader } from "./common";
import { CompactModelCapabilityBadges } from "./modelCapabilities";
import { ToggleSwitch } from "./settingsControls";
import { toolIcon } from "./toolUi";
import {
  defaultToolPreferences,
  modelSupportsToolUse,
  toolPreferenceEnabled,
  updateToolPreference
} from "./toolPreferences";
import type {
  AvailableTool,
  ChatToolPreferences,
  ComposerAttachment,
  ModelInfo,
  ThinkingMode
} from "./types";
import { dismissMobileKeyboard, usesTouchViewport } from "./viewport";

export function StartChatComposer({
  isBusy,
  isDisabled,
  isGenerating = false,
  placeholder,
  selectedModelInfo,
  availableTools = [],
  toolPreferences,
  thinkingMode = "auto",
  warning,
  onToolPreferencesChange,
  onThinkingModeChange,
  onStop,
  onUploadAttachment,
  onRemoveAttachment,
  onSubmit,
  autoFocusOnReady = true
}: {
  isBusy: boolean;
  isDisabled: boolean;
  isGenerating?: boolean;
  placeholder: string;
  selectedModelInfo?: ModelInfo | null;
  availableTools?: AvailableTool[];
  toolPreferences?: ChatToolPreferences;
  thinkingMode?: ThinkingMode;
  warning?: string | null;
  onToolPreferencesChange?: (preferences: ChatToolPreferences) => void | Promise<void>;
  onThinkingModeChange?: (mode: ThinkingMode) => void;
  onStop?: () => void;
  onUploadAttachment?: (file: File) => Promise<ComposerAttachment> | ComposerAttachment;
  onRemoveAttachment?: (attachment: ComposerAttachment) => Promise<void>;
  onSubmit: (
    prompt: string,
    attachments?: ComposerAttachment[],
    toolPreferences?: ChatToolPreferences,
    thinkMode?: ThinkingMode
  ) => Promise<void>;
  autoFocusOnReady?: boolean;
}) {
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const [isToolMenuOpen, setIsToolMenuOpen] = useState(false);
  const [isThinkingMenuOpen, setIsThinkingMenuOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const toolMenuRef = useRef<HTMLDivElement | null>(null);
  const thinkingMenuRef = useRef<HTMLDivElement | null>(null);
  const canAttach = Boolean(onUploadAttachment);
  const currentToolPreferences = toolPreferences ?? defaultToolPreferences;
  const canUseTools =
    Boolean(onToolPreferencesChange) &&
    availableTools.length > 0 &&
    modelSupportsToolUse(selectedModelInfo);
  const canControlThinking = Boolean(onThinkingModeChange && selectedModelInfo?.supports_thinking);
  const thinkingOptions = thinkingModeOptionsForModel(selectedModelInfo);
  const activeThinkingMode = normalizedThinkingModeForModel(thinkingMode, selectedModelInfo);
  const enabledToolCount = availableTools.filter((tool) =>
    toolPreferenceEnabled(currentToolPreferences, tool.id)
  ).length;
  const showPartialToolBadge =
    currentToolPreferences.tool_use_enabled && enabledToolCount < availableTools.length;
  const hasUploadingAttachment = attachments.some((attachment) => attachment.status === "uploading");
  const hasUnsupportedImageWarning =
    selectedModelInfo !== undefined &&
    selectedModelInfo !== null &&
    !selectedModelInfo.supports_images &&
    attachments.some(isImageAttachment);
  const visibleWarning =
    warning ?? (hasUnsupportedImageWarning ? "Images may not be supported by this model." : null);
  const canSubmit =
    prompt.trim().length > 0 && !isDisabled && !hasUploadingAttachment && (!isBusy || isGenerating);
  const thinkingModeLabel = thinkingModeOptionLabel(activeThinkingMode);
  const composerClassName = [
    "chat-composer",
    !canAttach ? "chat-composer-no-attach" : "",
    !canUseTools ? "chat-composer-no-tools" : "",
    !canControlThinking ? "chat-composer-no-thinking" : ""
  ]
    .filter(Boolean)
    .join(" ");

  useEffect(() => {
    if (autoFocusOnReady && !isGenerating && !usesTouchViewport()) {
      textareaRef.current?.focus();
    }
  }, [autoFocusOnReady, isGenerating]);

  useEffect(() => {
    if (!canAttach) {
      setAttachments([]);
    }
  }, [canAttach]);

  useEffect(() => {
    if (!canUseTools) {
      setIsToolMenuOpen(false);
    }
  }, [canUseTools]);

  useEffect(() => {
    if (!isToolMenuOpen && !isThinkingMenuOpen) {
      return;
    }

    function closeOnOutsidePointer(event: MouseEvent | TouchEvent) {
      const target = event.target;
      const inToolMenu = target instanceof Node && toolMenuRef.current?.contains(target);
      const inThinkingMenu = target instanceof Node && thinkingMenuRef.current?.contains(target);
      if (!inToolMenu && !inThinkingMenu) {
        setIsToolMenuOpen(false);
        setIsThinkingMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", closeOnOutsidePointer);
    document.addEventListener("touchstart", closeOnOutsidePointer);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsidePointer);
      document.removeEventListener("touchstart", closeOnOutsidePointer);
    };
  }, [isToolMenuOpen, isThinkingMenuOpen]);

  function updateTools(nextPreferences: ChatToolPreferences) {
    void onToolPreferencesChange?.(nextPreferences);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    const submittedPrompt = prompt;
    const submittedAttachments = attachments.filter(
      (attachment) => attachment.status === "ready" || attachment.status === "uploaded"
    );
    setPrompt("");
    setAttachments([]);
    const shouldRestoreFocus = !usesTouchViewport();
    if (!shouldRestoreFocus) {
      textareaRef.current?.blur();
      dismissMobileKeyboard();
    }
    await onSubmit(
      submittedPrompt,
      submittedAttachments,
      currentToolPreferences,
      activeThinkingMode
    );
    if (shouldRestoreFocus) {
      window.requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  async function uploadFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!onUploadAttachment || files.length === 0) {
      return;
    }

    for (const file of files) {
      const pendingAttachment = pendingComposerAttachment(newPendingAttachmentId(), file);

      setAttachments((current) => [...current, pendingAttachment]);

      try {
        const attachment = await onUploadAttachment(file);
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingAttachment.id ? attachment : currentAttachment
          )
        );
      } catch (uploadError) {
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingAttachment.id
              ? {
                  ...currentAttachment,
                  status: "error",
                  error: uploadError instanceof Error ? uploadError.message : "Upload failed"
                }
              : currentAttachment
          )
        );
      }
    }
  }

  async function removeAttachment(attachment: ComposerAttachment) {
    if (attachment.status === "uploaded" && !attachment.isExisting && onRemoveAttachment) {
      try {
        await onRemoveAttachment(attachment);
      } catch (removeError) {
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === attachment.id
              ? {
                  ...currentAttachment,
                  error: removeError instanceof Error ? removeError.message : "Delete failed"
                }
              : currentAttachment
          )
        );
        return;
      }
    }

    setAttachments((current) =>
      current.filter((currentAttachment) => currentAttachment.id !== attachment.id)
    );
  }

  return (
    <>
      {visibleWarning && <p className="composer-warning">{visibleWarning}</p>}
      <form className={composerClassName} onSubmit={submit}>
        {attachments.length > 0 && (
          <AttachmentChipList
            attachments={attachments}
            onRemove={(attachment) => void removeAttachment(attachment)}
          />
        )}
        {canAttach && (
          <div className="composer-attach">
            <input
              ref={fileInputRef}
              className="visually-hidden"
              type="file"
              multiple
              accept={attachmentAcceptTypes}
              onChange={uploadFiles}
            />
            <button
              type="button"
              aria-label="Attach files"
              title="Attach files"
              disabled={isDisabled}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip />
            </button>
          </div>
        )}
        {canUseTools && (
          <div className="composer-tools" ref={toolMenuRef}>
            <button
              type="button"
              aria-label="Tool settings"
              title="Tool settings"
              className={
                currentToolPreferences.tool_use_enabled
                  ? "composer-tool-button active"
                  : "composer-tool-button composer-tool-button-off"
              }
              disabled={isDisabled}
              onClick={() => {
                setIsThinkingMenuOpen(false);
                setIsToolMenuOpen((open) => !open);
              }}
            >
              <Wrench />
              {!currentToolPreferences.tool_use_enabled && (
                <span className="tool-off-indicator" aria-hidden="true">
                  <X />
                </span>
              )}
              {showPartialToolBadge && (
                <span className="tool-count-indicator" aria-hidden="true">
                  {enabledToolCount}/{availableTools.length}
                </span>
              )}
            </button>
            {isToolMenuOpen && (
              <div className="composer-tool-menu">
                <ToggleSwitch
                  label="Tool use"
                  description="Allow this chat to use enabled tools."
                  checked={currentToolPreferences.tool_use_enabled}
                  compact
                  onChange={(checked) =>
                    updateTools({
                      ...currentToolPreferences,
                      tool_use_enabled: checked
                    })
                  }
                />
                <div
                  className={
                    currentToolPreferences.tool_use_enabled
                      ? "composer-tool-list"
                      : "composer-tool-list disabled"
                  }
                >
                  {availableTools.map((tool) => (
                    <ToggleSwitch
                      key={tool.id}
                      icon={toolIcon(tool.id)}
                      label={tool.label}
                      description={tool.description}
                      checked={toolPreferenceEnabled(currentToolPreferences, tool.id)}
                      disabled={!currentToolPreferences.tool_use_enabled}
                      compact
                      onChange={(checked) =>
                        updateTools(
                          updateToolPreference(currentToolPreferences, tool.id, checked)
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {canControlThinking && (
          <div className="composer-thinking" ref={thinkingMenuRef}>
            <button
              type="button"
              aria-label={`Thinking effort: ${thinkingModeLabel}`}
              title={`Thinking effort: ${thinkingModeLabel}`}
              className={`composer-thinking-button active thinking-mode-${activeThinkingMode}`}
              disabled={isDisabled}
              onClick={() => {
                setIsToolMenuOpen(false);
                setIsThinkingMenuOpen((open) => !open);
              }}
            >
              <Brain />
              {activeThinkingMode === "false" ? (
                <span className="thinking-off-indicator" aria-hidden="true">
                  <X />
                </span>
              ) : activeThinkingMode === "auto" ? (
                <span className="thinking-auto-indicator" aria-hidden="true">
                  A
                </span>
              ) : (
                <span className={`thinking-effort-indicator ${activeThinkingMode}`} aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              )}
            </button>
            {isThinkingMenuOpen && (
              <div className="composer-thinking-menu">
                {thinkingOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={
                      activeThinkingMode === option.value
                        ? "composer-thinking-option active"
                        : "composer-thinking-option"
                    }
                    onClick={() => {
                      onThinkingModeChange?.(option.value);
                      setIsThinkingMenuOpen(false);
                    }}
                  >
                    <span>{option.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        <textarea
          ref={textareaRef}
          rows={3}
          value={prompt}
          disabled={isDisabled || (isBusy && !isGenerating)}
          placeholder={placeholder}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        {selectedModelInfo && (
          <div className="composer-model-capabilities">
            <CompactModelCapabilityBadges
              model={selectedModelInfo}
              hideTools={canUseTools}
              hideThinking={canControlThinking}
            />
          </div>
        )}
        <div className="composer-actions">
          {isGenerating && (
            <button type="button" aria-label="Stop generation" onClick={onStop}>
              <Square />
            </button>
          )}
          <button type="submit" aria-label="Send message" disabled={!canSubmit}>
            {isBusy && !isGenerating ? <RetroLoader /> : <SendHorizontal />}
          </button>
        </div>
      </form>
    </>
  );
}

const thinkingModeOptions: Array<{ value: ThinkingMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "false", label: "Off" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" }
];

function thinkingModeOptionsForModel(model?: ModelInfo | null) {
  if (!model || supportsThinkOff(model)) {
    return thinkingModeOptions;
  }

  return thinkingModeOptions.filter((option) => option.value !== "false");
}

function normalizedThinkingModeForModel(mode: ThinkingMode, model?: ModelInfo | null) {
  const options = thinkingModeOptionsForModel(model);
  return options.some((option) => option.value === mode) ? mode : "auto";
}

function thinkingModeOptionLabel(mode: ThinkingMode) {
  return thinkingModeOptions.find((option) => option.value === mode)?.label ?? "Auto";
}

function supportsThinkOff(model: ModelInfo) {
  return !model.name.toLocaleLowerCase().includes("gpt-oss");
}
