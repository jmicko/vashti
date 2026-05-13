import {
  ChangeEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  ChevronLeft,
  ChevronRight,
  Copy,
  Paperclip,
  Pencil,
  RefreshCw,
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
import type {
  ChatMessage,
  ComposerAttachment,
  ImageOpenHandler,
  MessageStreamSegment,
  ModelInfo,
  VersionInfo
} from "./types";
import { dismissMobileKeyboard } from "./viewport";

export function MessageBubble({
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
  selectedModelInfo
}: {
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
  selectedModelInfo?: ModelInfo | null;
}) {
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
  const orderedSegments = streamSegments?.length ? streamSegments : storedOrderedSegments;
  const hasOrderedSegments = orderedSegments.length > 0;
  const attachments = activeMessageAttachments(message);
  const [isEditing, setIsEditing] = useState(false);
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

  return (
    <article
      className={`message-bubble message-bubble-${message.role}`}
      data-message-id={message.id}
    >
      <div className="message-header">
        <p className="message-role">{messageLabel(message)}</p>
        {versionInfo && (
          <VersionSwitcher
            versionInfo={versionInfo}
            isBusy={isBusy}
            isGenerating={isGenerating}
          />
        )}
      </div>
      {!hasOrderedSegments && hasThinkingDetail && !message.is_deleted && (
        <details className="message-thinking">
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
        />
      ) : content ? (
        <MarkdownContent content={content} />
      ) : (
        <p>{message.status === "streaming" ? <RetroLoader /> : "No content"}</p>
      )}
      {!isEditing && (
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
        </div>
      )}
    </article>
  );
}

function VersionSwitcher({
  versionInfo,
  isBusy,
  isGenerating
}: {
  versionInfo: VersionInfo;
  isBusy: boolean;
  isGenerating: boolean;
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
        disabled={isBusy || isGenerating || hidePrevious}
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
        disabled={isBusy || isGenerating || hideNext}
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
