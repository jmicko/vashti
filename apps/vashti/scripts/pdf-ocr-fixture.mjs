import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";

const stream = (value, options = "") => {
  const data = Buffer.from(value);
  return Buffer.concat([Buffer.from(`<< /Length ${data.length} ${options} >>\nstream\n`), data, Buffer.from("\nendstream")]);
};
const objects = [
  "<< /Type /Catalog /Pages 2 0 R >>",
  "",
  "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 550 700] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
  "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  stream("BT /F1 25 Tf 40 615 Td (Synthetic mixed PDF - embedded text.) Tj ET")
];
const pages = [3];
for (const [code, pixelRotation, pageRotation] of [
  ["HARBOR-4739", 0, 0],
  ["BIRCH-2816", 90, 0],
  ["MARBLE-6952", 270, 0],
  ["ORCHID-5724", 180, 0],
  ["SUMMIT-3487", 90, 270],
  ["RIVER-7149", 0, 0]
]) {
  const lines = `WATERFALL SUPPLY\nShipping receipt\n\nReference number ${code}\nDispatch date 15 October\nDestination: North Pier\nApproved by Sam Turner`;
  const image = execFileSync("convert", [
    "-size", "1100x1400", "xc:white", "-font", "DejaVu-Sans",
    "-pointsize", "48", "-fill", "black", "-annotate", "+65+165", lines,
    "-rotate", String(pixelRotation), "-colorspace", "Gray", "-quality", "92", "jpeg:-"
  ]);
  const [width, height] = pixelRotation % 180 ? [700, 550] : [550, 700];
  const page = objects.length + 1;
  pages.push(page);
  objects.push(
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] /Rotate ${pageRotation} /Resources << /XObject << /Scan ${page + 2} 0 R >> >> /Contents ${page + 1} 0 R >>`,
    stream(`q ${width} 0 0 ${height} 0 0 cm /Scan Do Q`),
    stream(image, `/Type /XObject /Subtype /Image /Width ${width * 2} /Height ${height * 2} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode`)
  );
}
objects[1] = `<< /Type /Pages /Kids [${pages.map(page => `${page} 0 R`).join(" ")}] /Count ${pages.length} >>`;
const data = [Buffer.from("%PDF-1.4\n")], offsets = [];
let offset = data[0].length;
for (const [index, object] of objects.entries()) {
  offsets.push(offset);
  const block = Buffer.concat([Buffer.from(`${index + 1} 0 obj\n`), Buffer.from(object), Buffer.from("\nendobj\n")]);
  data.push(block);
  offset += block.length;
}
const xref = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n` + offsets.map(
  value => `${String(value).padStart(10, "0")} 00000 n \n`
).join("");
data.push(Buffer.from(`${xref}trailer\n<< /Root 1 0 R /Size ${objects.length + 1} >>\nstartxref\n${offset}\n%%EOF\n`));
const destination = process.argv[2] ?? "/tmp/vashti-ocr-fixture.pdf";
writeFileSync(destination, Buffer.concat(data), { mode: 0o600 });
console.log(`Synthetic mixed PDF: ${destination}`);
