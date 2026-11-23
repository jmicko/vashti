import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import type { Plugin } from "vite";

export function pdfAssets(): Plugin {
  const require = createRequire(import.meta.url);
  const packageFile = require.resolve("pdfjs-dist/package.json");
  const root = dirname(packageFile);
  const prefix = `pdf-runtime/${require(packageFile).version}/`;
  const pdfFiles = new Map([
    ["pdf.worker.mjs", join(root, "legacy/build/pdf.worker.min.mjs")],
    ["PDFjs-LICENSE.txt", join(root, "LICENSE")],
    ["Yjs-LICENSE.txt", join(dirname(require.resolve("yjs/package.json")), "LICENSE")]
  ]);
  for (const folder of ["standard_fonts", "cmaps", "iccs"]) {
    for (const entry of readdirSync(join(root, folder), { withFileTypes: true })) {
      if (entry.isFile()) pdfFiles.set(`${folder}/${entry.name}`, join(root, folder, entry.name));
    }
  }

  const ocrPackage = require.resolve("tesseract.js/package.json");
  const ocrRoot = dirname(ocrPackage);
  const core = dirname(createRequire(ocrPackage).resolve("tesseract.js-core/package.json"));
  const english = dirname(require.resolve("@tesseract.js-data/eng/package.json"));
  const ocrPrefix = `ocr-runtime/${require(ocrPackage).version}/`;
  const ocrFiles = new Map([
    ["worker.min.js", join(ocrRoot, "dist/worker.min.js")],
    ["eng.traineddata.gz", join(english, "4.0.0_best_int/eng.traineddata.gz")],
    ["Tesseract-LICENSE.txt", join(ocrRoot, "LICENSE.md")],
    ["Tesseract-core-LICENSE.txt", join(core, "LICENSE")]
  ]);
  for (const file of ["tesseract-core-lstm", "tesseract-core-simd-lstm", "tesseract-core-relaxedsimd-lstm"]) {
    ocrFiles.set(file + ".wasm.js", join(core, file + ".wasm.js"));
  }
  const files = new Map([
    ...Array.from(pdfFiles, ([key, path]) => [prefix + key, path] as const),
    ...Array.from(ocrFiles, ([key, path]) => [ocrPrefix + key, path] as const)
  ]);

  return {
    name: "vashti-pdf-runtime",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!request.url?.startsWith("/" + prefix) && !request.url?.startsWith("/" + ocrPrefix)) return next();
        const key = request.url.slice(1).split("?")[0];
        const file = files.get(key);
        if (!file) { response.statusCode = 404; return response.end(); }
        response.setHeader("Content-Type", /\.m?js$/.test(key) ? "text/javascript" : "application/octet-stream");
        if (key === ocrPrefix + "worker.min.js") {
          response.setHeader("Content-Security-Policy", "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'");
        }
        response.end(readFileSync(file));
      });
    },
    generateBundle() {
      for (const [key, path] of files) {
        this.emitFile({ type: "asset", fileName: key, source: readFileSync(path) });
      }
    }
  };
}
