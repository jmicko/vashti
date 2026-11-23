import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, Download, Minus, Plus, Search, X } from "lucide-react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { readAttachmentBlob } from "../attachments";
import { requestJson } from "../api";
import { saveFile } from "../saveFile";
import type { AttachmentInfo } from "../types";
import { isPdfBlob, loadPdf, pdfPageText } from "./reader";
import { PDF_PREVIEW_BYTES, type PdfText } from "./types";
import "./viewer.css";

export default function PdfViewer({ attachment, initialPage, onClose }: {
  attachment: AttachmentInfo; initialPage: number; onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const original = useRef<Blob | null>(null);
  const scanText = useRef<Promise<PdfText | null> | null>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy>();
  const [page, setPage] = useState(initialPage);
  const [zoom, setZoom] = useState(1);
  const [width, setWidth] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [saving, setSaving] = useState(false);
  const [password, setPassword] = useState("");
  const [unlock, setUnlock] = useState<{ submit: (password: string) => void; retry: boolean }>();
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [matches, setMatches] = useState<number[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [hit, setHit] = useState(0);

  useEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => { element.close(); };
  }, []);

  useEffect(() => {
    const element = viewport.current!;
    const observer = new ResizeObserver(() => setWidth(element.clientWidth));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let active = true;
    let task: ReturnType<typeof loadPdf> | undefined;
    void (async () => {
      if (attachment.size_bytes > PDF_PREVIEW_BYTES) throw new Error("This PDF is too large to preview. Download it to open it.");
      const blob = await readAttachmentBlob(attachment);
      if (!active) return;
      original.current = blob;
      if (!await isPdfBlob(blob)) throw new Error("This attachment is not a valid PDF.");
      if (blob.size > PDF_PREVIEW_BYTES) throw new Error("This PDF is too large to preview. Download it to open it.");
      const data = await blob.arrayBuffer();
      if (!active) return;
      task = loadPdf(data);
      task.onPassword = (submit: (password: string) => void, reason: number) => {
        if (active) setUnlock({ submit, retry: reason === 2 });
      };
      const document = await task.promise;
      if (!active) return;
      setPage(Math.min(initialPage, document.numPages));
      setDoc(document);
    })().catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "Could not open PDF.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => {
      active = false;
      original.current = null;
      scanText.current = null;
      void task?.destroy().catch(() => undefined);
    };
  }, [attachment, initialPage]);

  useEffect(() => {
    if (!doc || !width) return;
    let active = true;
    let task: { cancel: () => void; promise: Promise<unknown> } | undefined;
    const surface = canvas.current!;
    void (async () => {
      const source = await doc.getPage(page);
      if (!active) return;
      const base = source.getViewport({ scale: 1 });
      const scale = Math.min(2, (width - 24) / Math.max(1, base.width)) * zoom;
      const view = source.getViewport({ scale });
      if (!Number.isFinite(view.width * view.height) || view.width <= 0 || view.height <= 0) throw new Error("Invalid page dimensions.");
      const ratio = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(12_000_000 / (view.width * view.height)), 8192 / Math.max(view.width, view.height));
      const staged = document.createElement("canvas");
      staged.width = Math.max(1, Math.floor(view.width * ratio));
      staged.height = Math.max(1, Math.floor(view.height * ratio));
      task = source.render({ canvas: staged, viewport: view, transform: [ratio, 0, 0, ratio, 0, 0] });
      await task.promise;
      if (!active) return;
      surface.width = staged.width;
      surface.height = staged.height;
      surface.style.width = `${Math.min(view.width, 8192)}px`;
      surface.getContext("2d")?.drawImage(staged, 0, 0);
      setError("");
      source.cleanup();
    })().catch(() => {
      if (active) setError("Could not render this PDF page.");
    });
    return () => { active = false; task?.cancel(); };
  }, [doc, page, width, zoom]);

  useEffect(() => {
    if (!doc || !search) return;
    let active = true;
    setSearching(true);
    setSearchError("");
    setMatches([]);
    setHit(0);
    const found: number[] = [];
    const normalize = (text: string) => text.toLocaleLowerCase().replace(/\s+/g, " ");
    const needle = normalize(search);
    void (async () => {
      scanText.current ??= attachment.pdf_text ? Promise.resolve(attachment.pdf_text)
        : attachment.pdf_has_text && !attachment.import_asset_id && !attachment.data_url
          ? requestJson<PdfText | null>(`/api/attachments/${attachment.id}/pdf-text`).catch(() => null)
          : Promise.resolve(null);
      const stored = await scanText.current;
      const recognized = new Map(stored?.pages.map(page => [page.page, page.text]));
      for (let n = 1; n <= doc.numPages && active; n++) {
        const source = await doc.getPage(n);
        const text = await pdfPageText(source, 200_000);
        if (!active) return;
        if (normalize(text).includes(needle) || normalize(recognized.get(n) ?? "").includes(needle)) {
          if (!found.length) setPage(n);
          found.push(n);
          setMatches([...found]);
        }
        if (n % 12 === 0) await new Promise(resolve => setTimeout(resolve, 0));
      }
    })().catch(() => {
      if (active) setSearchError("Could not search the entire PDF.");
    }).finally(() => {
      if (active) setSearching(false);
    });
    return () => { active = false; };
  }, [doc, search, attachment]);

  function go(next: number) {
    setPage(next);
    viewport.current?.scrollTo(0, 0);
  }

  function goHit(next: number) {
    setHit(next);
    go(matches[next]);
  }

  async function download() {
    setSaving(true);
    setActionError("");
    try {
      const blob = original.current ?? await readAttachmentBlob(attachment);
      await saveFile(blob, attachment.original_filename, "application/pdf", true);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "Could not save attachment.");
    } finally {
      setSaving(false);
    }
  }

  return createPortal(
    <dialog className="pdf-viewer" ref={dialog} aria-label={attachment.original_filename}
      onCancel={event => { event.preventDefault(); onClose(); }}
      onKeyDown={event => {
        if (!doc || event.target instanceof HTMLInputElement) return;
        if (event.key === "ArrowLeft" && page > 1) go(page - 1);
        if (event.key === "ArrowRight" && page < doc.numPages) go(page + 1);
      }}>
      <header className="pdf-title">
        <strong title={attachment.original_filename}>{attachment.original_filename}</strong>
        <button type="button" aria-label="Find in PDF" title="Find in PDF" disabled={!doc}
          aria-expanded={searchOpen} onClick={() => {
            setSearchOpen(!searchOpen);
            if (searchOpen) { setSearch(""); setSearching(false); }
          }}><Search /></button>
        <button type="button" aria-label="Download PDF" title="Download PDF" disabled={saving} onClick={() => void download()}><Download /></button>
        <button type="button" aria-label="Close PDF" title="Close PDF" onClick={onClose}><X /></button>
      </header>
      {searchOpen && <form className="pdf-find" onSubmit={event => { event.preventDefault(); setSearch(query.trim()); }}>
        <input type="search" autoFocus aria-label="Search PDF text" placeholder="Find in PDF" value={query}
          onChange={event => { setQuery(event.target.value); if (!event.target.value) { setSearch(""); setSearching(false); } }} />
        <button type="submit" aria-label="Search PDF" title="Search PDF"><Search /></button>
        <button type="button" aria-label="Previous matching page" title="Previous match" disabled={!hit || !search}
          onClick={() => goHit(hit - 1)}><ChevronLeft /></button>
        <button type="button" aria-label="Next matching page" title="Next match" disabled={hit + 1 >= matches.length || !search}
          onClick={() => goHit(hit + 1)}><ChevronRight /></button>
        {search && <small role="status">{searchError || (searching ? "Searching..." :
          matches.length ? `${matches.length} matching ${matches.length === 1 ? "page" : "pages"} (${hit + 1}/${matches.length})` : "No matches")}</small>}
      </form>}
      <div className="pdf-pages" ref={viewport}>
        {unlock && <form className="pdf-unlock" onSubmit={event => {
          event.preventDefault(); unlock.submit(password); setUnlock(undefined); setPassword("");
        }}>
          <label>{unlock.retry ? "Incorrect PDF password" : "PDF password"}
            <input type="password" autoFocus value={password} onChange={event => setPassword(event.target.value)} required />
          </label>
          <button type="submit">Unlock</button>
        </form>}
        {loading && !unlock && <p role="status">Opening PDF...</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <canvas hidden={!doc || Boolean(error)} ref={canvas} aria-label={`Page ${page}`} />
      </div>
      {actionError && <p className="pdf-action-error error" role="alert">{actionError}</p>}
      {doc && <footer className="pdf-controls">
        <button type="button" aria-label="Previous page" title="Previous page" disabled={page <= 1} onClick={() => go(page - 1)}><ChevronLeft /></button>
        <span aria-live="polite">Page {page} / {doc.numPages}</span>
        <button type="button" aria-label="Next page" title="Next page" disabled={page >= doc.numPages} onClick={() => go(page + 1)}><ChevronRight /></button>
        <button type="button" aria-label="Zoom out" title="Zoom out" disabled={zoom <= 0.75} onClick={() => setZoom(z => z - 0.25)}><Minus /></button>
        <button type="button" aria-label="Fit width" title="Fit width" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
        <button type="button" aria-label="Zoom in" title="Zoom in" disabled={zoom >= 3} onClick={() => setZoom(z => z + 0.25)}><Plus /></button>
      </footer>}
    </dialog>, document.body
  );
}
