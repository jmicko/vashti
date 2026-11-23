import { readFile, writeFile } from "node:fs/promises";
import { syntheticExport } from "../web/src/imports/fixtures.mjs";

const destination = process.argv[2] ?? "/tmp/vashti-chatgpt-import-fixture.zip";
const image = await readFile(new URL("../src/images/fixtures/64.png", import.meta.url));
const zip = await syntheticExport(image, process.argv[3]);
await writeFile(destination, new Uint8Array(await zip.arrayBuffer()), { mode: 0o600 });
console.log(`Synthetic ChatGPT export fixture: ${destination} (${zip.size} bytes)`);
