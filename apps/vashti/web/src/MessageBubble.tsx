import {
  ChangeEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
  type SyntheticEvent
} from "react";
import {
  CircleAlert,
  ChevronLeft,
  ChevronRight,
  Copy,
  Info,
  LoaderCircle,
  Paperclip,
  Pencil,
  RefreshCw,
  StepForward,
  Save,
  SendHorizontal,
  Trash2,
  X
} from "lucide-react";
import {
  AttachmentChipList,
  MessageAttachments,
  attachmentAcceptTypes,
  cleanupDraftUploads,
  isImageAttachment,
  newPendingAttachmentId,
  pendingComposerAttachment,
  submittableAttachments
} from "./attachments";
import {
  activeMessageAttachments,
  composerAttachmentFromExisting,
  messageDisplaySegmentsFromRevision,
  parseThinkingText
} from "./chatMessages";
import { RetroLoader } from "./common";
import { MarkdownContent } from "./MarkdownContent";
import { MessageStreamContent, ThinkingContent, thinkingSummary } from "./messageContent";
import { ModelAvatar } from "./ModelAvatar";
import { MessageVersionCarousel } from "./MessageVersionCarousel";
import type {
  ChatMessage,
  ComposerAttachment,
  HostedPendingSend,
  ImageOpenHandler,
  MessageStats,
  MessageStreamSegment,
  ModelInfo,
  VersionInfo
} from "./types";
import { dismissMobileKeyboard } from "./viewport";

export function PendingOutgoingMessage({
  pendingSend,
  onDiscard,
  onImageOpen,
  onRetry
}: {
  pendingSend: HostedPendingSend;
  onDiscard: () => void;
  onImageOpen?: ImageOpenHandler;
  onRetry: () => void;
}) {
  const isSending = pendingSend.status === "sending";

  return (
    <article
      className={
        isSending
          ? "message-bubble message-bubble-user message-bubble-pending"
          : "message-bubble message-bubble-user message-bubble-pending failed"
      }
      data-message-id={`pending-${pendingSend.id}`}
      aria-busy={isSending}
      aria-live="polite"
    >
      <MessageAttachments attachments={pendingSend.attachments} onImageOpen={onImageOpen} />
      <MarkdownContent content={pendingSend.prompt} />
      <div className={isSending ? "pending-send-status" : "pending-send-status failed"}>
        {isSending ? <LoaderCircle className="pending-send-spinner" /> : <CircleAlert />}
        <div>
          <strong>{isSending ? "Sending..." : "Message not sent"}</strong>
          {!isSending && pendingSend.error_text && <span>{pendingSend.error_text}</span>}
        </div>
      </div>
      {!isSending && (
        <div className="message-actions pending-send-actions">
          <button type="button" onClick={onRetry}>
            <RefreshCw />
            Retry
          </button>
          <button type="button" className="danger-button" onClick={onDiscard}>
            <Trash2 />
            Discard
          </button>
        </div>
      )}
    </article>
  );
}

type MessageAvatarInfo = {
  displayName: string;
  assetId?: string | null;
  privateAssetId?: string | null;
  cropX?: number;
  cropY?: number;
  cropSize?: number;
};

type MessageBubbleProps = {
  message: ChatMessage;
  versionInfo: VersionInfo | null;
  copied: boolean;
  isBusy: boolean;
  isGenerating: boolean;
  canBranch?: boolean;
  canRegenerate?: boolean;
  canEdit?: boolean;
  streamSegments?: MessageStreamSegment[];
  thinkingDurationSeconds: number | null;
  onCopy: (message: ChatMessage) => Promise<void>;
  onDelete: (message: ChatMessage) => void;
  onBranch: (
    message: ChatMessage,
    contentText: string,
    attachments?: ComposerAttachment[]
  ) => Promise<void>;
  onEdit: (
    message: ChatMessage,
    contentText: string,
    attachments?: ComposerAttachment[]
  ) => Promise<void>;
  onImageOpen?: ImageOpenHandler;
  onRemoveAttachment?: (attachment: ComposerAttachment) => Promise<void>;
  onUploadAttachment?: (file: File) => Promise<ComposerAttachment> | ComposerAttachment;
  onRegenerate: (message: ChatMessage) => Promise<void>;
  onContinue: (message: ChatMessage) => Promise<void>;
  selectedModelInfo?: ModelInfo | null;
  modelAvatar?: MessageAvatarInfo | null;
  modelAvatarForMessage?: (message: ChatMessage) => MessageAvatarInfo | null;
  streamSegmentsForMessage?: (message: ChatMessage) => MessageStreamSegment[] | undefined;
  thinkingDurationForMessage?: (message: ChatMessage) => number | null;
  isCarouselPreview?: boolean;
};

export function MessageBubble(props: MessageBubbleProps) {
  const { message, versionInfo } = props;
  const bubble = <MessageBubbleCard {...props} />;

  if (!versionInfo) {
    return bubble;
  }

  return (
    <MessageVersionCarousel
      isBusy={props.isBusy}
      role={message.role}
      versionInfo={versionInfo}
      renderVersion={(version, index) => {
        const previewMessage = messageAtVersion(version);

        return (
          <MessageBubbleCard
            {...props}
            message={previewMessage}
            versionInfo={versionInfoAtIndex(versionInfo, index)}
            copied={false}
            isBusy={false}
            isGenerating={previewMessage.status === "streaming"}
            streamSegments={
              props.streamSegmentsForMessage?.(previewMessage) ??
              (previewMessage.id === message.id ? props.streamSegments : undefined)
            }
            thinkingDurationSeconds={
              props.thinkingDurationForMessage?.(previewMessage) ??
              (previewMessage.id === message.id ? props.thinkingDurationSeconds : null)
            }
            modelAvatar={
              props.modelAvatarForMessage
                ? props.modelAvatarForMessage(previewMessage)
                : props.modelAvatar
            }
            isCarouselPreview
          />
        );
      }}
    >
      {bubble}
    </MessageVersionCarousel>
  );
}

function MessageBubbleCard({
  message,
  versionInfo,
  copied,
  isBusy,
  isGenerating,
  canBranch = true,
  canRegenerate = true,
  canEdit = true,
  streamSegments,
  thinkingDurationSeconds,
  onCopy,
  onDelete,
  onBranch,
  onEdit,
  onImageOpen,
  onRemoveAttachment,
  onUploadAttachment,
  onRegenerate,
  onContinue,
  selectedModelInfo,
  modelAvatar,
  isCarouselPreview = false
}: MessageBubbleProps) {
  const content = message.is_deleted
    ? "Message deleted"
    : message.active_revision?.content_text.trim() || "";
  const parsedThinking = useMemo(
    () => parseThinkingText(message.active_revision?.thinking_text ?? ""),
    [message.active_revision?.thinking_text]
  );
  const storedOrderedSegments = useMemo(
    () =>
      messageDisplaySegmentsFromRevision(
        message.active_revision?.content_text ?? "",
        message.active_revision?.thinking_text ?? ""
      ),
    [message.active_revision?.content_text, message.active_revision?.thinking_text]
  );
  const thinking = parsedThinking.thinkingText.trim();
  const hasThinkingDetail =
    thinking || parsedThinking.segments.some((segment) => segment.type === "tool");
  const shouldUseStreamSegments = message.status === "streaming" && Boolean(streamSegments?.length);
  const orderedSegments = shouldUseStreamSegments ? streamSegments ?? [] : storedOrderedSegments;
  const hasOrderedSegments = orderedSegments.length > 0;
  const attachments = activeMessageAttachments(message);
  const [isEditing, setIsEditing] = useState(false);
  const [isThinkingOpen, setIsThinkingOpen] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [draft, setDraft] = useState(content);
  const [draftAttachments, setDraftAttachments] = useState<ComposerAttachment[]>([]);
  const editFileInputRef = useRef<HTMLInputElement | null>(null);
  const hasUnsupportedImageWarning =
    selectedModelInfo !== undefined &&
    selectedModelInfo !== null &&
    !selectedModelInfo.supports_images &&
    draftAttachments.some(isImageAttachment);
  const hasUploadingAttachment = draftAttachments.some(
    (attachment) => attachment.status === "uploading"
  );

  useEffect(() => {
    if (!isEditing) {
      setDraft(content);
    }
  }, [content, isEditing]);

  function handleThinkingToggle(event: SyntheticEvent<HTMLDetailsElement>) {
    setIsThinkingOpen(event.currentTarget.open);
  }

  async function saveEdit() {
    dismissMobileKeyboard();
    await onEdit(message, draft, submittableAttachments(draftAttachments));
    setIsEditing(false);
  }

  async function sendEdit() {
    dismissMobileKeyboard();
    setIsEditing(false);
    await onBranch(message, draft, submittableAttachments(draftAttachments));
  }

  function startEditing() {
    setDraft(content);
    setDraftAttachments(attachments.map(composerAttachmentFromExisting));
    setIsEditing(true);
  }

  async function cancelEdit() {
    dismissMobileKeyboard();
    await cleanupDraftUploads(draftAttachments, onRemoveAttachment);
    setIsEditing(false);
  }

  async function addEditFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!onUploadAttachment || files.length === 0) {
      return;
    }

    for (const file of files) {
      const pendingId = newPendingAttachmentId();
      const pendingAttachment = pendingComposerAttachment(pendingId, file);
      setDraftAttachments((current) => [...current, pendingAttachment]);

      try {
        const attachment = await onUploadAttachment(file);
        setDraftAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId ? attachment : currentAttachment
          )
        );
      } catch (uploadError) {
        setDraftAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId
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

  async function removeDraftAttachment(attachment: ComposerAttachment) {
    if (attachment.status === "uploaded" && !attachment.isExisting && onRemoveAttachment) {
      try {
        await onRemoveAttachment(attachment);
      } catch (removeError) {
        setDraftAttachments((current) =>
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

    setDraftAttachments((current) =>
      current.filter((currentAttachment) => currentAttachment.id !== attachment.id)
    );
  }

  const showMessageLabel = message.role !== "user";
  const showMessageHeader = showMessageLabel || Boolean(versionInfo);

  return (
    <article
      className={`message-bubble message-bubble-${message.role}`}
      data-message-id={isCarouselPreview ? undefined : message.id}
      data-carousel-message-id={isCarouselPreview ? message.id : undefined}
    >
      {showMessageHeader && (
        <div className={showMessageLabel ? "message-header" : "message-header message-header-end"}>
          {showMessageLabel && (
            <div className="message-role-wrap">
              {modelAvatar && (
                <ModelAvatar
                  displayName={modelAvatar.displayName}
                  assetId={modelAvatar.assetId}
                  privateAssetId={modelAvatar.privateAssetId}
                  cropX={modelAvatar.cropX}
                  cropY={modelAvatar.cropY}
                  cropSize={modelAvatar.cropSize}
                  className="model-avatar-message"
                />
              )}
              <p className="message-role">{messageLabel(message)}</p>
            </div>
          )}
          {versionInfo && (
            <VersionSwitcher versionInfo={versionInfo} isBusy={isBusy} />
          )}
        </div>
      )}
      {!hasOrderedSegments && hasThinkingDetail && !message.is_deleted && (
        <details
          className="message-thinking"
          open={isThinkingOpen}
          onToggle={handleThinkingToggle}
        >
          <summary>{thinkingSummary(message, thinkingDurationSeconds)}</summary>
          <ThinkingContent segments={parsedThinking.segments} />
        </details>
      )}
      {!isEditing && !message.is_deleted && attachments.length > 0 && (
        <MessageAttachments attachments={attachments} onImageOpen={onImageOpen} />
      )}
      {isEditing ? (
        <div className="message-edit">
          {hasUnsupportedImageWarning && (
            <p className="composer-warning">Images may not be supported by this model.</p>
          )}
          {draftAttachments.length > 0 && (
            <AttachmentChipList
              attachments={draftAttachments}
              onRemove={(attachment) => void removeDraftAttachment(attachment)}
            />
          )}
          {onUploadAttachment && (
            <div className="message-edit-attach">
              <input
                ref={editFileInputRef}
                className="visually-hidden"
                type="file"
                multiple
                accept={attachmentAcceptTypes}
                onChange={addEditFiles}
              />
              <button
                type="button"
                className="secondary-button"
                disabled={isBusy}
                onClick={() => editFileInputRef.current?.click()}
              >
                <Paperclip />
                <span>Attach</span>
              </button>
            </div>
          )}
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} rows={5} />
          <div className="message-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={isBusy}
              onClick={() => void cancelEdit()}
            >
              <X />
              <span>Cancel</span>
            </button>
            <button
              type="button"
              disabled={isBusy || hasUploadingAttachment || draft.trim() === ""}
              onClick={() => void saveEdit()}
            >
              <Save />
              <span>{isBusy ? "Saving..." : "Save"}</span>
            </button>
            {message.role === "user" && canBranch && (
              <button
                type="button"
                disabled={isBusy || isGenerating || hasUploadingAttachment || draft.trim() === ""}
                onClick={() => void sendEdit()}
              >
                <SendHorizontal />
                <span>Send</span>
              </button>
            )}
          </div>
        </div>
      ) : hasOrderedSegments ? (
        <MessageStreamContent
          message={message}
          segments={orderedSegments}
          thinkingDurationSeconds={thinkingDurationSeconds}
          isThinkingOpen={isThinkingOpen}
          onThinkingToggle={handleThinkingToggle}
        />
      ) : content ? (
        <MarkdownContent content={content} />
      ) : (
        <p>{message.status === "streaming" ? <RetroLoader /> : "No content"}</p>
      )}
      {message.status !== "streaming" && message.error_text && (
        <div className="message-generation-notice" role="status">
          <CircleAlert />
          <span>{message.error_text}</span>
        </div>
      )}
      {!isEditing && (
        <>
          <div className="message-actions">
            {message.role === "assistant" && canRegenerate && (
              <button
                type="button"
                className="message-icon-button"
                title="Regenerate"
                aria-label="Regenerate"
                disabled={isBusy || isGenerating || message.status === "streaming"}
                onClick={() => {
                  dismissMobileKeyboard();
                  void onRegenerate(message);
                }}
              >
                <RefreshCw />
              </button>
            )}
            {message.role === "assistant" && (
              <button
                type="button"
                className="message-icon-button"
                title="Continue response"
                aria-label="Continue response"
                disabled={
                  isBusy ||
                  isGenerating ||
                  message.is_deleted ||
                  message.status === "streaming" ||
                  content === ""
                }
                onClick={() => {
                  dismissMobileKeyboard();
                  void onContinue(message);
                }}
              >
                <StepForward />
              </button>
            )}
            <button
              type="button"
              className="message-icon-button"
              title="Copy"
              aria-label="Copy"
              disabled={message.is_deleted || content === ""}
              onClick={() => void onCopy(message)}
            >
              <Copy />
              {copied && <span>Copied</span>}
            </button>
            <button
              type="button"
              className="message-icon-button"
              title="Edit"
              aria-label="Edit"
              disabled={isBusy || !canEdit || message.is_deleted || message.status === "streaming"}
              onClick={startEditing}
            >
              <Pencil />
            </button>
            <button
              type="button"
              className="message-icon-button danger-button"
              title="Delete"
              aria-label="Delete"
              disabled={isBusy || message.status === "streaming"}
              onClick={() => onDelete(message)}
            >
              <Trash2 />
            </button>
            {message.role === "assistant" && (
              <button
                type="button"
                className={
                  message.stats
                    ? "message-icon-button message-stats-button"
                    : "message-icon-button message-stats-button missing"
                }
                title={message.stats ? "Message stats" : "No stats received"}
                aria-label={message.stats ? "Message stats" : "No stats received"}
                aria-expanded={showStats}
                disabled={isGenerating || message.status === "streaming"}
                onClick={() => setShowStats((open) => !open)}
              >
                <Info />
              </button>
            )}
          </div>
          {message.role === "assistant" && showStats && (
            <MessageStatsPanel stats={message.stats ?? null} />
          )}
        </>
      )}
    </article>
  );
}

function messageAtVersion(version: VersionInfo["versions"][number]): ChatMessage {
  return {
    ...version.message,
    active_revision_id: version.revision.id,
    active_revision: version.revision
  };
}

function versionInfoAtIndex(versionInfo: VersionInfo, index: number): VersionInfo {
  return {
    ...versionInfo,
    index,
    canPrevious: index > 0,
    canNext: index < versionInfo.total - 1,
    onPrevious: () => versionInfo.onSelectIndex(index - 1),
    onNext: () => versionInfo.onSelectIndex(index + 1)
  };
}

function MessageStatsPanel({ stats }: { stats: MessageStats | null }) {
  if (!stats) {
    return <div className="message-stats-panel muted">No stats received for this message.</div>;
  }

  const tokensPerSecond =
    stats.eval_count && stats.eval_duration
      ? stats.eval_count / (stats.eval_duration / 1_000_000_000)
      : null;

  return (
    <div className="message-stats-panel">
      <StatItem label="Speed" value={tokensPerSecond ? `${tokensPerSecond.toFixed(1)} tok/s` : "n/a"} />
      <StatItem label="Output" value={formatCount(stats.eval_count, "token")} />
      <StatItem label="Context" value={formatCount(stats.prompt_eval_count, "token")} />
      <StatItem label="Total" value={formatNanoseconds(stats.total_duration)} />
      <StatItem label="Load" value={formatNanoseconds(stats.load_duration)} />
      <StatItem label="Prompt eval" value={formatNanoseconds(stats.prompt_eval_duration)} />
      <StatItem label="Eval" value={formatNanoseconds(stats.eval_duration)} />
    </div>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <strong>{label}</strong>
      {value}
    </span>
  );
}

function formatCount(value: number | null | undefined, unit: string) {
  if (value === null || value === undefined) {
    return "n/a";
  }

  return `${value.toLocaleString()} ${unit}${value === 1 ? "" : "s"}`;
}

function formatNanoseconds(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "n/a";
  }

  if (value < 1_000_000) {
    return `${Math.round(value / 1_000).toLocaleString()} us`;
  }

  const seconds = value / 1_000_000_000;
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

function VersionSwitcher({
  versionInfo,
  isBusy
}: {
  versionInfo: VersionInfo;
  isBusy: boolean;
}) {
  const hidePrevious = !versionInfo.canPrevious;
  const hideNext = !versionInfo.canNext;

  return (
    <div className="version-switcher" aria-label="Message version selector">
      <button
        type="button"
        className={
          hidePrevious
            ? "message-icon-button version-arrow-hidden"
            : "message-icon-button"
        }
        title={hidePrevious ? undefined : "Previous version"}
        aria-label="Previous version"
        aria-hidden={hidePrevious}
        tabIndex={hidePrevious ? -1 : undefined}
        disabled={isBusy || hidePrevious}
        onClick={versionInfo.onPrevious}
      >
        <ChevronLeft />
      </button>
      <span className="version-count">
        {versionInfo.index + 1}/{versionInfo.total}
      </span>
      <button
        type="button"
        className={
          hideNext ? "message-icon-button version-arrow-hidden" : "message-icon-button"
        }
        title={hideNext ? undefined : "Next version"}
        aria-label="Next version"
        aria-hidden={hideNext}
        tabIndex={hideNext ? -1 : undefined}
        disabled={isBusy || hideNext}
        onClick={versionInfo.onNext}
      >
        <ChevronRight />
      </button>
    </div>
  );
}

function messageLabel(message: ChatMessage) {
  if (message.role === "assistant") {
    return message.persona_name_snapshot ?? message.model_name ?? "assistant";
  }

  return message.role;
}
