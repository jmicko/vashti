import { createWorker, OEM } from "tesseract.js";
import type { Worker } from "tesseract.js";
import { version } from "tesseract.js/package.json";
import type { PDFPageProxy } from "pdfjs-dist";
import { withinDeadline } from "./deadline";

export async function startPdfOcr(signal?: AbortSignal) {
  signal?.throwIfAborted();
  const base = new URL(`/ocr-runtime/${version}/`, window.location.origin).href;
  let worker: Worker | undefined;
  let closed = false;
  let failInitialization: (cause: unknown) => void = () => {};
  const errors = new Promise<never>((_, reject) => { failInitialization = reject; });
  const pending = createWorker("eng", OEM.LSTM_ONLY, {
    workerPath: base + "worker.min.js",
    corePath: base,
    langPath: base,
    workerBlobURL: false,
    cacheMethod: "none",
    errorHandler: cause => { if (!worker) failInitialization(cause); }
  }).then(value => {
    if (closed) void value.terminate();
    else worker = value;
    return value;
  });
  try {
    const recognizer = await withinDeadline(Promise.race([pending, errors]), 35_000, signal);
    return {
      async recognize(page: PDFPageProxy) {
        signal?.throwIfAborted();
        const baseView = page.getViewport({ scale: 1 });
        const scale = Math.min(3, 1800 / Math.max(baseView.width, baseView.height));
        const angles = baseView.width > baseView.height ? [270, 90, 0, 180] : [0, 180, 270, 90];
        let best = { text: "", confidence: 0 };
        for (const angle of angles) {
          signal?.throwIfAborted();
          const view = page.getViewport({ scale, rotation: (page.rotate + angle) % 360 });
          if (!(view.width > 0 && view.height > 0)) throw new Error("Invalid PDF page size.");
          const canvas = document.createElement("canvas");
          canvas.width = Math.ceil(view.width);
          canvas.height = Math.ceil(view.height);
          const render = page.render({ canvas, viewport: view, background: "#ffffff" });
          try {
            await withinDeadline(render.promise, 15_000, signal);
            // Chromium's PNG encoder can wait for idle time past our deadline in WebView.
            // WebP at quality 1 is lossless there and encodes off the main thread.
            const image = await withinDeadline(new Promise<Blob>((resolve, reject) => {
              canvas.toBlob(blob => {
                if (blob) resolve(blob);
                else reject(new Error("Could not read the scanned page."));
              }, "image/webp", 1);
            }), 10_000, signal);
            const { data } = await withinDeadline(recognizer.recognize(image, { rotateAuto: true }), 35_000, signal);
            const text = data.text.trim();
            if (text && (data.confidence > best.confidence || !best.text)) best = { text, confidence: data.confidence };
            if (best.confidence >= 78 && best.text.length >= 10) break;
          } finally {
            render.cancel();
            canvas.width = canvas.height = 0;
          }
        }
        return { text: best.confidence >= 45 ? best.text : "", partial: Boolean(best.text) && best.confidence < 65 };
      },
      close: () => recognizer.terminate()
    };
  } catch (cause) {
    closed = true;
    await worker?.terminate();
    throw cause;
  }
}
