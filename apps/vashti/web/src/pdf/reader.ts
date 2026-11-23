import { getDocument, GlobalWorkerOptions, VerbosityLevel, version } from "pdfjs-dist/legacy/build/pdf.mjs";
import type { PDFPageProxy } from "pdfjs-dist";
import { withinDeadline } from "./deadline";
import {
  PDF_CONTEXT_PAGES, PDF_PAGE_TEXT_BYTES, PDF_PREVIEW_BYTES, PDF_STORED_TEXT_BYTES,
  PDF_UPLOAD_BYTES, type PdfPreparation, type PdfText
} from "./types";

export async function isPdfBlob(blob: Blob) {
  const header = await blob.slice(0, 8).text();
  return /^%PDF-\d\.\d/.test(header);
}

export function loadPdf(data: ArrayBuffer) {
  if (data.byteLength > PDF_PREVIEW_BYTES) throw new Error("This PDF is too large to preview.");
  const base = typeof document === "undefined" ? undefined : `/pdf-runtime/${version}/`;
  if (base) GlobalWorkerOptions.workerSrc = base + "pdf.worker.mjs";
  return getDocument({
    data: new Uint8Array(data),
    useWasm: false,
    enableXfa: false,
    useWorkerFetch: false,
    ...(base && { cMapUrl: base + "cmaps/", standardFontDataUrl: base + "standard_fonts/", iccUrl: base + "iccs/" }),
    maxImageSize: 16_000_000,
    canvasMaxAreaInBytes: 48_000_000,
    verbosity: VerbosityLevel.ERRORS
  });
}

function cleanPageText(text: string) {
  return text.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "").trim();
}

export async function pdfPageText(page: PDFPageProxy, maxChars = PDF_PAGE_TEXT_BYTES) {
  const reader = page.streamTextContent().getReader();
  const chunks: string[] = [];
  let length = 0;
  try {
    while (length < maxChars) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const item of value.items) {
        if (!("str" in item)) continue;
        const text = item.str.slice(0, maxChars - length);
        chunks.push(text, item.hasEOL ? "\n" : " ");
        length += text.length + 1;
        if (length >= maxChars) break;
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
  return cleanPageText(chunks.join(""));
}

export async function extractPdfText(file: File, { signal, onProgress }: PdfPreparation = {}): Promise<PdfText> {
  signal?.throwIfAborted();
  if (file.size > PDF_UPLOAD_BYTES) throw new Error("Attached PDFs must be 8 MiB or smaller.");
  if (!await isPdfBlob(file)) throw new Error("This file is not a valid PDF.");
  const data = await withinDeadline(file.arrayBuffer(), 20_000, signal);
  signal?.throwIfAborted();
  const task = loadPdf(data);
  // Password-protected documents may be saved without any extracted text.
  task.onPassword = () => { void task.destroy().catch(() => undefined); };
  const result: PdfText = { pages: [], truncated: false };
  const encoder = new TextEncoder();
  let ocr: Awaited<ReturnType<typeof import("./ocr").startPdfOcr>> | undefined;
  let ocrUnavailable = typeof document === "undefined";
  try {
    const doc = await withinDeadline(task.promise, 20_000, signal);
    const last = Math.min(doc.numPages, PDF_CONTEXT_PAGES);
    const pageBytes = Math.min(PDF_PAGE_TEXT_BYTES, Math.floor(PDF_STORED_TEXT_BYTES / last));
    result.truncated = doc.numPages > last;
    for (let n = 1; n <= last; n++) {
      signal?.throwIfAborted();
      onProgress?.(`Reading PDF · ${n}/${last}`);
      const page = await withinDeadline(doc.getPage(n), 15_000, signal);
      try {
        let text = await withinDeadline(pdfPageText(page, pageBytes + 1), 15_000, signal);
        if (!ocrUnavailable && text.replace(/[^\p{L}\p{N}]/gu, "").length < 10) {
          onProgress?.(`Recognizing text · ${n}/${last}`);
          try {
            if (!ocr) ocr = await (await import("./ocr")).startPdfOcr(signal);
            const scanned = await ocr.recognize(page);
            const recognized = cleanPageText(scanned.text);
            result.truncated ||= scanned.partial;
            if (recognized.length > text.length) text = recognized;
          } catch (cause) {
            signal?.throwIfAborted();
            console.warn("PDF OCR could not read a page", cause instanceof Error ? cause.name : "WorkerError");
            ocrUnavailable = !ocr;
            await ocr?.close().catch(() => undefined);
            ocr = undefined;
            result.truncated = true;
          }
        }
        const bytes = encoder.encode(text);
        if (bytes.length > pageBytes) {
          text = new TextDecoder().decode(bytes.subarray(0, pageBytes)).replace(/\ufffd$/, "").trimEnd();
          result.truncated = true;
        }
        if (text) result.pages.push({ page: n, text });
      } finally {
        page.cleanup();
      }
    }
  } catch {
    signal?.throwIfAborted();
    result.truncated = true;
  } finally {
    await ocr?.close().catch(() => undefined);
    await task.destroy().catch(() => undefined);
  }
  return result;
}
