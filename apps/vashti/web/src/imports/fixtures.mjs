import { BlobWriter, TextReader, Uint8ArrayReader, ZipWriter } from "@zip.js/zip.js/lib/zip-core-native.js";
import * as Y from "yjs";

export function syntheticConversation(id = "fixture-conversation") {
  const node = (role, content, parent, extra = {}) => ({ parent, message: { author: { role }, create_time: 100, content, metadata: { model_slug: "original-model" }, ...extra } });
  return { id, title: "Synthetic import fixture", create_time: 100, update_time: 104, current_node: "alternative", is_archived: true, mapping: {
    root: { parent: null, message: null },
    user: node("user", { content_type: "multimodal_text", parts: ["A synthetic prompt", { content_type: "image_asset_pointer", asset_pointer: "sediment://file-fixture" }] }, "root"),
    thought: node("assistant", { content_type: "thoughts", thoughts: [{ content: "Synthetic reasoning" }] }, "user"),
    recap: node("assistant", { content_type: "reasoning_recap", content: "Thought briefly" }, "thought"),
    first: node("assistant", { content_type: "text", parts: ["**First answer**"] }, "recap"),
    alternative: node("assistant", { content_type: "text", parts: ["Alternative answer"] }, "user")
  } };
}

export async function fixtureZip(files) {
  const writer = new ZipWriter(new BlobWriter(), { useWebWorkers: false });
  for (const [name, value] of files) await writer.add(name, typeof value === "string" ? new TextReader(value) : new Uint8ArrayReader(value));
  return writer.close();
}

export function syntheticPdf() {
  const stream = (text) => {
    const content = text ? `BT /F1 22 Tf 40 300 Td (${text}) Tj ET\n` : "0 0 80 80 re f\n";
    return `<< /Length ${content.length} >>\nstream\n${content}endstream`;
  };
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R 5 0 R 8 0 R] /Count 3 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 420] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
    stream("First page: synthetic apples"),
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 420] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
    stream("Second page: synthetic mangoes"),
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 420] /Contents 9 0 R >>",
    stream("")
  ];
  let result = "%PDF-1.4\n";
  const offsets = objects.map((object, index) => {
    const offset = result.length;
    result += `${index + 1} 0 obj\n${object}\nendobj\n`;
    return offset;
  });
  // The large comment keeps the ZIP fixture useful for chunked upload tests.
  result += "%" + " ".repeat(700_000) + "\n";
  const xref = result.length;
  result += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets) result += `${String(offset).padStart(10, "0")} 00000 n \n`;
  result += `trailer\n<< /Root 1 0 R /Size ${objects.length + 1} >>\nstartxref\n${xref}\n%%EOF\n`;
  return new TextEncoder().encode(result);
}

export function syntheticPageRecords(id, title) {
  const doc = new Y.Doc();
  const byId = doc.getMap("blocksById"), order = doc.getArray("topLevelBlockIds");
  for (const [key, text] of [["one", "## First section"], ["two", "A **formatted** paragraph.\n\n- A list item"]]) {
    const block = new Y.Map();
    block.set("kind", "markdown");
    block.set("markdown", new Y.Text(text));
    byId.set(key, block);
    order.push([key]);
  }
  const update = Buffer.from(Y.encodeStateAsUpdate(doc)).toString("base64url");
  doc.destroy();
  const identity = { page_id: id, through_sequence: 2 };
  return {
    page: {
      head: { page_id: id, stream_kind: "content", current_sequence: 2, checkpoint_id: "fixture-checkpoint" },
      includes_current_sequence: true,
      projection: { page_id: id, projection_through_sequence: 2, title, materialized_search_text: "First section\n\nA formatted paragraph.\n\nA list item", updated_at: "2026-11-01T17:00:00Z" }
    },
    checkpoint: {
      checkpoint: { ...identity, stream_kind: "content", checkpoint_id: "fixture-checkpoint" },
      document: { ...identity, format: "openai.page.yjs", schema_version: 2, yjs_state_update: update }
    }
  };
}

export async function syntheticExport(image, suffix = "") {
  const active = syntheticConversation("synthetic-import-ui-active" + suffix);
  active.title = "Synthetic import QA - branches and files"; active.is_archived = false; active.current_node = "first";
  active.mapping.user.message.metadata.attachments = [{ id: "file-document", name: "synthetic-report.pdf" }];
  active.mapping.first.message.content.parts = ["**Synthetic imported answer**\n\n- Markdown list\n- [Download synthetic report](sandbox:/mnt/data/synthetic-report.pdf)\n\n```rust\nfn synthetic_example() {}\n```\n\nA source \ue200cite\ue202fixture\ue201"];
  active.mapping.first.message.metadata.content_references = [{ url: "https://example.com/" }];
  const archived = syntheticConversation("synthetic-import-ui-archived" + suffix); archived.title = "Synthetic import QA - archived";
  archived.mapping.user.message.content = { content_type: "text", parts: ["An archived synthetic question"] };
  const id = "fixture-page" + suffix;
  const { page, checkpoint } = syntheticPageRecords(id, "Synthetic workspace Page" + (suffix ? " " + suffix : ""));
  return fixtureZip([
    ["conversations-000.json", JSON.stringify([active])], ["conversations-001.json", JSON.stringify([archived])],
    ["file-fixture.dat", image], ["file-document.dat", syntheticPdf()],
    ["conversation_asset_file_names.json", JSON.stringify({ "file-fixture.dat": "synthetic-image.png" })],
    [`pages/${id}/content/current.json`, JSON.stringify(page)],
    [`pages/${id}/content/checkpoints/fixture-checkpoint.json`, JSON.stringify(checkpoint)]
  ]);
}
