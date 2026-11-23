import {
  lazy, Suspense,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from "react";
import type { ReactNode } from "react";
import { ChevronLeft, ChevronRight, Download, FileText, Image as ImageIcon, X } from "lucide-react";
import { requestBlob, requestMultipartJson } from "./api";
import { PhotoViewer } from "./PhotoViewer";
import { saveImage } from "./saveImage";
import { saveFile } from "./saveFile";
import { RetroLoader } from "./common";
import { ImageGenerationSettings } from "./ImageGenerationSettings";
import { unixTimestamp, readPrivateImportAsset } from "./privateChatStore";
import { apiAssetUrl } from "./runtime";
import { isPdfAttachment, type PdfPreparation, type PdfText } from "./pdf/types";
import type {
  AttachmentInfo,
  AttachmentResponse,
  ComposerAttachment,
  ImageOpenHandler
} from "./types";

const PdfViewer = lazy(() => import("./pdf/PdfViewer"));

export const attachmentAcceptTypes = [
  "application/pdf",
  ".pdf",
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "text/*",
  ".txt",
  ".md",
  ".markdown",
  ".json",
  ".csv",
  ".log",
  ".toml",
  ".yaml",
  ".yml",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".css",
  ".html",
  ".rs",
  ".py",
  ".go",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".php",
  ".rb",
  ".sh"
].join(",");

export function AttachmentChipList({
  attachments,
  onRemove
}: {
  attachments: ComposerAttachment[];
  onRemove: (attachment: ComposerAttachment) => void;
}) {
  return (
    <div className="composer-attachments" aria-label="Attached files">
      {attachments.map((attachment) => (
        <ComposerAttachmentItem
          key={attachment.id}
          attachment={attachment}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}

function ComposerAttachmentItem({
  attachment,
  onRemove
}: {
  attachment: ComposerAttachment;
  onRemove: (attachment: ComposerAttachment) => void;
}) {
  const imageUrl = useAttachmentImageUrl(attachment);
  const isImage = isImageAttachment(attachment);
  const className = [
    "attachment-preview",
    attachment.status === "error" ? "attachment-chip-error" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className} title={attachment.error ?? attachment.original_filename}>
      <div className={isImage ? "attachment-preview-image" : "attachment-preview-image attachment-preview-document"}>
        {isImage && imageUrl ? (
          <img src={imageUrl} alt="" />
        ) : (
          <>
            {isImage ? <ImageIcon /> : <FileText />}
            {!isImage && <span className="attachment-type-label">{attachmentTypeLabel(attachment)}</span>}
          </>
        )}
      </div>
      <span>{attachment.original_filename}</span>
      <small title={attachment.progress || undefined}>{attachment.status === "uploading" ? (attachment.progress || <RetroLoader />) : attachment.pdf_has_text === false
        ? "No extractable text" : attachment.pdf_truncated ? "Partial text" : formatBytes(attachment.size_bytes)}</small>
      <button
        type="button"
        className="message-icon-button"
        aria-label={`Remove ${attachment.original_filename}`}
        onClick={() => onRemove(attachment)}
      >
        <X />
      </button>
    </div>
  );
}

export function MessageAttachments({
  attachments,
  onImageOpen
}: {
  attachments: AttachmentInfo[];
  onImageOpen?: ImageOpenHandler;
}) {
  if (attachments.length === 0) {
    return null;
  }

  const imageAttachments = attachments.filter(isImageAttachment);
  const fileAttachments = attachments.filter((attachment) => !isImageAttachment(attachment));

  return (
    <div className="message-attachments" aria-label="Message attachments">
      {imageAttachments.length > 0 && (
        <MessageImageCarousel attachments={imageAttachments} onImageOpen={onImageOpen} />
      )}
      {fileAttachments.map(attachment => <DocumentAttachment key={attachment.id} attachment={attachment} />)}
    </div>
  );
}

export function DocumentAttachment({ attachment, inlineLabel, pdfPage = 1 }: { attachment: AttachmentInfo; inlineLabel?: ReactNode; pdfPage?: number }) {
  const [saving, setSaving] = useState(false), [error, setError] = useState("");
  const [preview, setPreview] = useState(false);
  const inline=inlineLabel!==undefined;
  const pdf = isPdfAttachment(attachment);
  return <span className={inline?"imported-file-link":undefined}><button type="button" className={inline?"imported-file-button":"attachment-chip"} disabled={saving} title={attachment.original_filename} onClick={() => {
    if (pdf) { setPreview(true); return; }
    setSaving(true); setError("");
    void readAttachmentBlob(attachment).then(blob => saveFile(blob, attachment.original_filename, attachment.mime_type, true)).catch(cause => setError(cause instanceof Error ? cause.message : "Could not save attachment")).finally(() => setSaving(false));
  }}>{inline ? (saving?"Saving...":inlineLabel) : <>{attachmentIcon(attachment)}<span>{attachment.original_filename}</span><small>{saving ? "Saving..." : formatBytes(attachment.size_bytes)}</small></>}</button>{error && <span role="alert" className="error">{error}</span>}{preview && <Suspense fallback={<span role="status">Opening PDF...</span>}><PdfViewer attachment={attachment} initialPage={pdfPage} onClose={() => setPreview(false)}/></Suspense>}</span>;
}

export async function readAttachmentBlob(attachment: AttachmentInfo) {
  if (attachment.import_asset_id) return readPrivateImportAsset(attachment.import_asset_id);
  if (attachment.data_url?.startsWith("data:application/pdf;base64,")) {
    const bytes = Uint8Array.from(atob(attachment.data_url.slice("data:application/pdf;base64,".length)), char => char.charCodeAt(0));
    return new Blob([bytes], { type: "application/pdf" });
  }
  if (attachment.data_url) return (await fetch(attachment.data_url)).blob();
  if (attachment.text_content !== undefined) return new Blob([attachment.text_content], { type: attachment.mime_type });
  return requestBlob(`/api/attachments/${attachment.id}`);
}

function AttachmentImage({ attachment, alt, onLoad }: { attachment: AttachmentInfo; alt?: string; onLoad?: () => void }) {
  const [url, setUrl] = useState<string | null>(null), [error, setError] = useState("");
  useEffect(() => {
    if (!attachment.import_asset_id) return;
    let cancelled = false, objectUrl: string | null = null;
    void readPrivateImportAsset(attachment.import_asset_id).then(blob => {
      if (!cancelled) { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); }
    }).catch(() => { if (!cancelled) setError("Attachment unavailable; resume its import"); });
    return () => { cancelled = true; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [attachment.import_asset_id]);
  return <img src={attachment.import_asset_id ? url ?? undefined : attachmentDisplayUrl(attachment)} alt={error || alt || attachment.original_filename} title={error || undefined} loading="lazy" draggable={false} onLoad={onLoad} />;
}

function MessageImageCarousel({
  attachments,
  onImageOpen
}: {
  attachments: AttachmentInfo[];
  onImageOpen?: ImageOpenHandler;
}) {
  const [carouselMetrics, setCarouselMetrics] = useState({
    hasOverflow: false,
    canPrevious: false,
    canNext: false,
    hiddenRightCount: 0
  });
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pointerStartXRef = useRef<number | null>(null);
  const didSwipeRef = useRef(false);
  const hasMultipleImages = attachments.length > 1;

  const updateCarouselMetrics = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const viewRight = viewport.scrollLeft + viewport.clientWidth;
    const children = Array.from(viewport.children) as HTMLElement[];
    const nextMetrics = {
      hasOverflow: viewport.scrollWidth > viewport.clientWidth + 1,
      canPrevious: viewport.scrollLeft > 1,
      canNext: viewport.scrollWidth - viewRight > 1,
      hiddenRightCount: children.filter(
        (child) => child.offsetLeft + child.offsetWidth > viewRight + 1
      ).length
    };
    setCarouselMetrics((current) =>
      current.hasOverflow === nextMetrics.hasOverflow &&
      current.canPrevious === nextMetrics.canPrevious &&
      current.canNext === nextMetrics.canNext &&
      current.hiddenRightCount === nextMetrics.hiddenRightCount
        ? current
        : nextMetrics
    );
  }, []);

  useLayoutEffect(() => {
    updateCarouselMetrics();

    const viewport = viewportRef.current;
    if (!viewport || typeof ResizeObserver === "undefined") {
      return;
    }

    const resizeObserver = new ResizeObserver(updateCarouselMetrics);
    resizeObserver.observe(viewport);
    for (const child of Array.from(viewport.children)) {
      resizeObserver.observe(child);
    }

    return () => resizeObserver.disconnect();
  }, [attachments, updateCarouselMetrics]);

  function showPrevious() {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const children = Array.from(viewport.children) as HTMLElement[];
    const currentFirstIndex = children.findIndex(
      (child) => child.offsetLeft + child.offsetWidth > viewport.scrollLeft + 1
    );
    const previous = children[Math.max((currentFirstIndex < 0 ? 0 : currentFirstIndex) - 1, 0)];
    viewport.scrollTo({ left: previous?.offsetLeft ?? 0, behavior: "smooth" });
  }

  function showNext() {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const children = Array.from(viewport.children) as HTMLElement[];
    const next = children.find((child) => child.offsetLeft > viewport.scrollLeft + 1);
    viewport.scrollTo({
      left: next?.offsetLeft ?? viewport.scrollWidth - viewport.clientWidth,
      behavior: "smooth"
    });
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    pointerStartXRef.current = event.clientX;
    didSwipeRef.current = false;
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const startX = pointerStartXRef.current;
    pointerStartXRef.current = null;
    if (startX === null || !hasMultipleImages || !carouselMetrics.hasOverflow) {
      return;
    }

    const deltaX = event.clientX - startX;
    if (Math.abs(deltaX) < 36) {
      return;
    }

    didSwipeRef.current = true;
    if (deltaX > 0) {
      showPrevious();
    } else {
      showNext();
    }
  }

  if (attachments.length === 0) {
    return null;
  }

  return (
    <div
      className="message-image-carousel"
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={() => {
        pointerStartXRef.current = null;
      }}
    >
      <div className="message-image-track" ref={viewportRef} onScroll={updateCarouselMetrics}>
        {attachments.map((attachment) => (
          <button
            key={attachment.id}
            type="button"
            className="message-image-button"
            aria-label={`Open ${attachment.original_filename}`}
            onClick={() => {
              if (didSwipeRef.current) {
                didSwipeRef.current = false;
                return;
              }
              onImageOpen?.(attachment, attachments);
            }}
          >
            <AttachmentImage attachment={attachment} onLoad={updateCarouselMetrics} />
          </button>
        ))}
      </div>
      {hasMultipleImages && carouselMetrics.hasOverflow && (
        <>
          {carouselMetrics.canPrevious && (
            <button
              type="button"
              className="message-image-nav message-image-nav-previous"
              aria-label="Previous image"
              onClick={showPrevious}
            >
              <ChevronLeft />
            </button>
          )}
          {carouselMetrics.canNext && (
            <button
              type="button"
              className="message-image-nav message-image-nav-next"
              aria-label="Next image"
              onClick={showNext}
            >
              <ChevronRight />
            </button>
          )}
          {carouselMetrics.hiddenRightCount > 0 && (
            <span className="message-image-count" aria-label={`${attachments.length} images`}>
              <ImageIcon />
              {attachments.length}
            </span>
          )}
        </>
      )}
    </div>
  );
}

export function ImageViewer({
  attachments,
  index,
  onClose,
  onIndexChange
}: {
  attachments: AttachmentInfo[];
  index: number;
  onClose: () => void;
  onIndexChange: (index: number) => void;
}) {
  const attachment = attachments[index] ?? attachments[0];
  const canPrevious = index > 0;
  const canNext = index < attachments.length - 1;
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  useEffect(() => setError(""), [attachment?.id]);
  if (!attachment) {
    return null;
  }
  async function download() {
    setSaving(true);
    setError("");
    try {
      const blob = await readAttachmentBlob(attachment);
      await saveImage(blob, attachment.original_filename.replace(/\.[^.]+$/, ""));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not save image"); }
    finally { setSaving(false); }
  }
  const neighbor = (value: AttachmentInfo | undefined) => value && <AttachmentImage attachment={value} alt="" />;
  return (
    <PhotoViewer id={attachment.id} title={`${index + 1} of ${attachments.length}`}
      onClose={onClose} onStep={(direction) => onIndexChange(Math.max(0, Math.min(attachments.length - 1, index + direction)))}
      canPrevious={canPrevious} canNext={canNext}
      previous={neighbor(attachments[index - 1])} next={neighbor(attachments[index + 1])}
      actions={<button type="button" aria-label="Download image" title="Download image" disabled={saving} onClick={() => void download()}><Download /></button>}
      more={attachment.mime_type === "image/png" && !attachment.data_url && !attachment.import_asset_id
        ? <ImageGenerationSettings key={attachment.id} attachmentId={attachment.id} /> : undefined}
      notice={error && <p role="alert">{error}</p>}>
      <AttachmentImage attachment={attachment} />
    </PhotoViewer>
  );
}

function attachmentIcon(attachment: Pick<AttachmentInfo, "attachment_kind" | "mime_type">) {
  return attachment.attachment_kind === "image" || attachment.mime_type.startsWith("image/") ? (
    <ImageIcon />
  ) : (
    <FileText />
  );
}

function attachmentTypeLabel(attachment: Pick<AttachmentInfo, "original_filename" | "mime_type">) {
  const extension = attachment.original_filename.split(".").pop()?.trim();
  if (extension && extension !== attachment.original_filename && extension.length <= 5) {
    return extension.toLocaleUpperCase();
  }

  if (attachment.mime_type.includes("json")) {
    return "JSON";
  }
  if (attachment.mime_type.includes("markdown")) {
    return "MD";
  }
  if (attachment.mime_type.startsWith("text/")) {
    return "TXT";
  }

  return "DOC";
}

function useAttachmentImageUrl(attachment: ComposerAttachment) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!isImageAttachment(attachment)) {
      setUrl(null);
      return;
    }

    if (attachment.data_url) {
      setUrl(attachment.data_url);
      return;
    }

    if (attachment.file) {
      const objectUrl = URL.createObjectURL(attachment.file);
      setUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }

    if (attachment.import_asset_id) {
      let alive = true, objectUrl: string | null = null;
      setUrl(null);
      void readPrivateImportAsset(attachment.import_asset_id).then(blob => {
        if (alive) { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); }
      }).catch(() => { if (alive) setUrl(null); });
      return () => { alive = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
    }

    if (attachment.status === "uploaded" && !attachment.id.startsWith("pending-")) {
      setUrl(attachmentUrl(attachment.id));
      return;
    }

    setUrl(null);
  }, [attachment]);

  return url;
}

function attachmentUrl(attachmentId: string) {
  return apiAssetUrl(`/api/attachments/${attachmentId}`);
}

function attachmentDisplayUrl(attachment: AttachmentInfo) {
  return attachment.data_url ?? attachmentUrl(attachment.id);
}

function attachmentDownloadUrl(attachment: AttachmentInfo) {
  if (attachment.data_url) {
    return attachment.data_url;
  }

  if (attachment.text_content !== undefined) {
    return `data:${attachment.mime_type || "text/plain"};charset=utf-8,${encodeURIComponent(
      attachment.text_content
    )}`;
  }

  return attachmentUrl(attachment.id);
}

async function preparePdf(file: File, options?: PdfPreparation): Promise<PdfText | undefined> {
  if (!isPdfAttachment({ mime_type: file.type, original_filename: file.name })) return;
  const { extractPdfText } = await import("./pdf/reader");
  return extractPdfText(file, options);
}

function pdfAttachmentFields(pdf_text: PdfText) {
  return {
    mime_type: "application/pdf",
    attachment_kind: "file",
    pdf_text,
    pdf_has_text: pdf_text.pages.length > 0,
    pdf_truncated: pdf_text.truncated
  };
}

export async function prepareLocalAttachment(file: File, options?: PdfPreparation): Promise<ComposerAttachment> {
  const pdf = await preparePdf(file, options);
  return {
    id: newPendingAttachmentId(),
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: file.type || "application/octet-stream",
    size_bytes: file.size,
    attachment_kind: file.type.startsWith("image/") ? "image" : "text",
    status: "ready",
    file,
    ...(pdf ? pdfAttachmentFields(pdf) : {})
  };
}

export async function preparePrivateAttachment(file: File, options?: PdfPreparation): Promise<ComposerAttachment> {
  const mimeType = file.type || mimeTypeFromFilename(file.name) || "application/octet-stream";
  const isImage =
    mimeType.startsWith("image/") &&
    ["image/png", "image/jpeg", "image/gif", "image/webp"].includes(mimeType);
  const baseAttachment = {
    id: newPendingAttachmentId(),
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: mimeType,
    size_bytes: file.size,
    created_at: unixTimestamp(),
    status: "ready" as const,
    file
  };

  const pdf = await preparePdf(file, options);
  options?.signal?.throwIfAborted();
  if (pdf) {
    const original = new File([file], file.name, { type: "application/pdf" });
    return { ...baseAttachment, ...pdfAttachmentFields(pdf), data_url: await readFileAsDataUrl(original) };
  }

  if (isImage) {
    return {
      ...baseAttachment,
      attachment_kind: "image",
      data_url: await readFileAsDataUrl(file)
    };
  }

  return {
    ...baseAttachment,
    attachment_kind: "text",
    text_content: await readFileAsUtf8(file)
  };
}

function mimeTypeFromFilename(filename: string) {
  const extension = filename.split(".").pop()?.toLocaleLowerCase();
  switch (extension) {
    case "md":
    case "markdown":
      return "text/markdown";
    case "json":
      return "application/json";
    case "csv":
      return "text/csv";
    case "txt":
    case "log":
    case "toml":
    case "yaml":
    case "yml":
    case "js":
    case "jsx":
    case "ts":
    case "tsx":
    case "css":
    case "html":
    case "rs":
    case "py":
    case "go":
    case "java":
    case "c":
    case "cpp":
    case "h":
    case "hpp":
    case "php":
    case "rb":
    case "sh":
      return "text/plain";
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "gif":
      return "image/gif";
    case "webp":
      return "image/webp";
    default:
      return null;
  }
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read image"));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Failed to read image"));
        return;
      }
      resolve(reader.result);
    };
    reader.readAsDataURL(file);
  });
}

async function readFileAsUtf8(file: File) {
  try {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const text = decoder.decode(await file.arrayBuffer());
    if (text.includes("\u0000")) {
      throw new Error("Unsupported binary file");
    }
    return text;
  } catch (error) {
    throw new Error(
      error instanceof Error && error.message === "Unsupported binary file"
        ? "Only image files and UTF-8 text files are supported."
        : "Only image files and UTF-8 text files are supported."
    );
  }
}

export async function uploadAttachmentToChat(chatId: string, file: File, pdfText?: PdfText, options?: PdfPreparation) {
  const formData = new FormData();
  formData.append("file", file);
  const document = pdfText ?? await preparePdf(file, options);
  options?.signal?.throwIfAborted();
  if (document) options?.onProgress?.("Uploading PDF...");
  if (document) formData.append("pdf_context", JSON.stringify(document));

  const response = await requestMultipartJson<AttachmentResponse>(
    `/api/chats/${chatId}/attachments`,
    formData,
    { method: "POST" }
  );
  return response.attachment;
}

export async function uploadComposerAttachments(
  chatId: string,
  attachments: ComposerAttachment[]
) {
  const uploadedAttachments: ComposerAttachment[] = [];

  for (const attachment of attachments) {
    if (attachment.status === "uploaded") {
      uploadedAttachments.push(attachment);
      continue;
    }

    if (attachment.status === "ready" && attachment.file) {
      uploadedAttachments.push({
        ...(await uploadAttachmentToChat(chatId, attachment.file, attachment.pdf_text)),
        status: "uploaded" as const
      });
    }
  }

  return uploadedAttachments;
}

export function attachmentReferences(attachments: ComposerAttachment[]) {
  return attachments
    .filter((attachment) => attachment.status === "uploaded")
    .map((attachment) => ({ id: attachment.id }));
}

export function submittableAttachments(attachments: ComposerAttachment[]) {
  return attachments.filter(
    (attachment) => attachment.status === "ready" || attachment.status === "uploaded"
  );
}

export function isImageAttachment(
  attachment: Pick<AttachmentInfo, "attachment_kind" | "mime_type">
) {
  return attachment.attachment_kind === "image" || attachment.mime_type.startsWith("image/");
}

export function newPendingAttachmentId() {
  return `pending-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function pendingComposerAttachment(id: string, file: File): ComposerAttachment {
  return {
    id,
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: file.type || "application/octet-stream",
    size_bytes: file.size,
    attachment_kind: file.type.startsWith("image/") ? "image" : "text",
    status: "uploading",
    file
  };
}

export async function cleanupDraftUploads(
  attachments: ComposerAttachment[],
  onRemoveAttachment: ((attachment: ComposerAttachment) => Promise<void>) | undefined
) {
  if (!onRemoveAttachment) {
    return;
  }

  await Promise.allSettled(
    attachments
      .filter((attachment) => attachment.status === "uploaded" && !attachment.isExisting)
      .map((attachment) => onRemoveAttachment(attachment))
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(kib < 10 ? 1 : 0)} KB`;
  }

  const mib = kib / 1024;
  return `${mib.toFixed(mib < 10 ? 1 : 0)} MB`;
}
