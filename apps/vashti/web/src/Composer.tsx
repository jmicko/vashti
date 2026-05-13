import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { Paperclip, SendHorizontal, Square, Wrench, X } from "lucide-react";
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
import type { AvailableTool, ChatToolPreferences, ComposerAttachment, ModelInfo } from "./types";
import { dismissMobileKeyboard, usesTouchViewport } from "./viewport";

export function StartChatComposer({
  isBusy,
  isDisabled,
  isGenerating = false,
  placeholder,
  selectedModelInfo,
  availableTools = [],
  toolPreferences,
  warning,
  onToolPreferencesChange,
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
  warning?: string | null;
  onToolPreferencesChange?: (preferences: ChatToolPreferences) => void | Promise<void>;
  onStop?: () => void;
  onUploadAttachment?: (file: File) => Promise<ComposerAttachment> | ComposerAttachment;
  onRemoveAttachment?: (attachment: ComposerAttachment) => Promise<void>;
  onSubmit: (
    prompt: string,
    attachments?: ComposerAttachment[],
    toolPreferences?: ChatToolPreferences
  ) => Promise<void>;
  autoFocusOnReady?: boolean;
}) {
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const [isToolMenuOpen, setIsToolMenuOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const toolMenuRef = useRef<HTMLDivElement | null>(null);
  const canAttach = Boolean(onUploadAttachment);
  const currentToolPreferences = toolPreferences ?? defaultToolPreferences;
  const canUseTools =
    Boolean(onToolPreferencesChange) &&
    availableTools.length > 0 &&
    modelSupportsToolUse(selectedModelInfo);
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
    if (!isToolMenuOpen) {
      return;
    }

    function closeOnOutsidePointer(event: MouseEvent | TouchEvent) {
      const target = event.target;
      if (target instanceof Node && !toolMenuRef.current?.contains(target)) {
        setIsToolMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", closeOnOutsidePointer);
    document.addEventListener("touchstart", closeOnOutsidePointer);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsidePointer);
      document.removeEventListener("touchstart", closeOnOutsidePointer);
    };
  }, [isToolMenuOpen]);

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
    await onSubmit(submittedPrompt, submittedAttachments, currentToolPreferences);
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
      <form
        className={canAttach ? "chat-composer" : "chat-composer chat-composer-no-attach"}
        onSubmit={submit}
      >
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
              onClick={() => setIsToolMenuOpen((open) => !open)}
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
            <CompactModelCapabilityBadges model={selectedModelInfo} hideTools={canUseTools} />
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
