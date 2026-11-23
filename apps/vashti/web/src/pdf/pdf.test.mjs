import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { syntheticPdf } from "../imports/fixtures.mjs";

const compiled = await build({
  stdin: {
    contents: `export * from "./pdf/reader"; export * from "./pdf/types";
      export { privatePromptMessages } from "./chatMessages";`,
    resolveDir: fileURLToPath(new URL("..", import.meta.url)),
    loader: "ts"
  },
  bundle: true, write: false, format: "esm", platform: "node",
  loader: { ".css": "empty" },
  plugins: [{
    name: "external-pdf-for-node",
    setup(builder) {
      builder.onResolve({ filter: /^pdfjs-dist\// }, args => ({
        path: import.meta.resolve(args.path), external: true
      }));
    }
  }]
});
const app = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString("base64")}`);

test("PDF extraction preserves real page numbers and does not invent text for scans", async () => {
  const fixture = new File([syntheticPdf()], "unknown-type.pdf");
  const extract = await app.extractPdfText(fixture);
  assert.deepEqual(extract, { pages: [
    { page: 1, text: "First page: synthetic apples" },
    { page: 2, text: "Second page: synthetic mangoes" }
  ], truncated: false });
  await assert.rejects(app.extractPdfText(new File(["Not a PDF"], "bad.pdf")), /not a valid PDF/);
});

test("private prompt uses extracted pages only from the selected ancestry", () => {
  const make = (id, parent, role, attachments = []) => ({
    id, parent_message_id: parent, role, active_revision_id: id + "-rev",
    active_revision: { content_text: id, thinking_text: "" },
    attachments
  });
  const file = { id: "pdf-1", revision_id: "root-rev", original_filename: "Test.pdf",
    mime_type: "application/pdf", attachment_kind: "file", data_url: "private bytes",
    pdf_text: { pages: [{ page: 2, text: "Sample page text" }], truncated: false } };
  const inactive = { ...file, revision_id: "old-rev", pdf_text: { pages: [{ page: 3, text: "Wrong revision" }] } };
  const imported = { ...file, id: "download-only", pdf_text: undefined };
  const messages = [
    make("root", null, "user", [file, inactive, imported]),
    make("unselected", null, "user", [{...file, revision_id: "unselected-rev", pdf_text: { pages: [{ page: 4, text: "Wrong branch" }] }}]),
    make("generation", "root", "assistant")
  ];
  const prompt = JSON.stringify(app.privatePromptMessages(messages, "generation"));
  assert.match(prompt, /Sample page text/);
  assert.match(prompt, /\/__vashti_pdf\/pdf-1\/2/);
  assert.doesNotMatch(prompt, /Wrong revision|Wrong branch|private bytes|download-only/);
  assert.deepEqual(app.messagePdfAttachments(messages.slice(0, 1)).map(f => f.id), ["pdf-1", "download-only"]);
  assert.deepEqual(app.pdfCitation("/__vashti_pdf/pdf-1/2"), { id: "pdf-1", page: 2 });
  assert.equal(app.pdfCitation("/__vashti_pdf/%2e%2e/2"), null);
  assert.equal(app.pdfCitation("/__vashti_pdf/pdf-1/0"), null);
  assert.equal(app.pdfCitation("https://example.invalid/__vashti_pdf/pdf-1/2"), undefined);
});

test("short PDFs stay whole; long PDFs prioritize late pages across follow-ups", () => {
  const short = {pages: [{page: 1, text: "Apple"}, {page: 2, text: "Mango"}], truncated: false};
  assert.equal(app.selectPdfText(short, "page 2", 72_000).document, short);
  const long = {
    pages: Array.from({length: 60}, (_, index) => ({
      page: index + 1,
      text: `Page ${index + 1} ` + "a synthetic fixture line ".repeat(105) +
        (index === 59 ? "Reference HARBOR-4739: North Pier." : "")
    })),
    truncated: false
  };
  for (const question of ["What is listed on page 60?", "Where does HARBOR dispatch?"]) {
    const selected = app.selectPdfText(long, question, 72_000);
    assert(selected.document.pages.some(page => page.page === 60 && /4739/.test(page.text)));
    assert(selected.document.truncated);
    assert(selected.used <= 24_000);
  }
  const overview = app.selectPdfText(long, "Summarize this document", 72_000);
  assert.equal(overview.document.pages.length, 60);
  assert(overview.used <= 24_000);
  assert.equal(app.selectPdfText(long, "page 60", 0).document.pages.length, 0);
});
