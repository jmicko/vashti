import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";

Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  window: new EventTarget(),
});
globalThis.FileReader = class {
  readAsDataURL(blob) {
    blob
      .arrayBuffer()
      .then((bytes) => {
        this.result = `data:${blob.type};base64,${Buffer.from(bytes).toString("base64")}`;
        this.onload?.();
      })
      .catch((e) => {
        this.error = e;
        this.onerror?.();
      });
  }
};
const clip = await readFile(
  new URL("../../../src/videos/fixtures/clip.mp4", import.meta.url),
);
const png = await readFile(
  new URL("../../../src/images/fixtures/64.png", import.meta.url),
);
const calls = [],
  exports = [];
let holdResult;
globalThis.__videoApi = {
  requestMultipartJson: async (path, form, options) => {
    assert.equal(path, "/api/videos/frames");
    assert.equal(options.method, "POST");
    assert.ok(form.get("image") instanceof Blob);
    return { id: "uploaded", width: 64, height: 64 };
  },
  requestJson: async (path, options = {}) => {
    calls.push({ path, input: options.body && JSON.parse(options.body) });
    if (path === "/api/videos/private")
      return { id: JSON.parse(options.body).request.request_id };
    throw new Error(`Unexpected fixture request ${path}`);
  },
  requestBlob: async (path) => {
    calls.push({ path });
    if (holdResult) await holdResult;
    return new Blob([clip], { type: "video/mp4" });
  },
};
const bundle = await build({
  stdin: {
    contents: `export * from './videos/repository'; export * from './videos/types'; export * from './privateChatStore'; export * from './saveFile';`,
    resolveDir: fileURLToPath(new URL("../", import.meta.url)),
    loader: "ts",
  },
  bundle: true,
  write: false,
  format: "esm",
  platform: "node",
  plugins: [
    {
      name: "offline-video",
      setup(build) {
        build.onResolve({ filter: /^\.\.?\/api$/ }, () => ({
          path: "api",
          namespace: "fixture",
        }));
        build.onResolve({ filter: /^\.\/runtime$/ }, () => ({
          path: "native",
          namespace: "fixture",
        }));
        build.onLoad({ filter: /.*/, namespace: "fixture" }, ({ path }) => ({
          contents:
            path === "api"
              ? `export const requestJson=(...a)=>globalThis.__videoApi.requestJson(...a); export const requestBlob=(...a)=>globalThis.__videoApi.requestBlob(...a); export const requestMultipartJson=(...a)=>globalThis.__videoApi.requestMultipartJson(...a);`
              : `export const isNativeRuntime=()=>true;export const invokeNative=async(name,args)=>{globalThis.__videoExports.push({name,args});return {saved:true}};`,
        }));
      },
    },
  ],
});
globalThis.__videoExports = exports;
const app = await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
);
const login = (id = "video-tests", server = "fixture") =>
  app.setPrivateStorageUser(
    id,
    {
      user_id: id,
      key_material: Buffer.alloc(32, id.charCodeAt(0)).toString("base64"),
    },
    server,
  );
async function source() {
  const id = crypto.randomUUID();
  await app.writePrivateImage(`video:frame:${id}`, {
    data: `data:image/png;base64,${png.toString("base64")}`,
    thumb: `data:image/png;base64,${png.toString("base64")}`,
  });
  return id;
}

test("server first-frame uploads explicitly use POST and release decoded pixels", async () => {
  login("upload");
  let closed = false;
  globalThis.createImageBitmap = async () => ({
    width: 64,
    height: 64,
    close() {
      closed = true;
    },
  });
  assert.equal(
    (
      await app
        .videoRepository("server")
        .frame(new File([png], "frame.png", { type: "image/png" }))
    ).id,
    "uploaded",
  );
  assert.equal(closed, true);
});

test("aspect presets stay on the advertised grid and never overrun the pixel budget", () => {
  const wf = {
    dimension_multiple: 32,
    min_dimension: 256,
    max_dimension: 1280,
    max_pixels: 452608,
  };
  for (const ratio of [
    1,
    5 / 4,
    4 / 5,
    4 / 3,
    3 / 4,
    3 / 2,
    2 / 3,
    16 / 10,
    10 / 16,
    16 / 9,
    9 / 16,
    2,
    0.5,
    21 / 9,
    9 / 21,
    3,
    1 / 3,
  ]) {
    const { width, height } = app.videoSize(ratio, wf);
    assert.equal(width % 32, 0);
    assert.equal(height % 32, 0);
    assert.ok(width >= 256 && height >= 256 && width <= 1280 && height <= 1280);
    assert.ok(width * height <= 452608);
    assert.ok(Math.abs(Math.log(width / height / ratio)) < 0.055);
  }
  assert.throws(() => app.videoSize(1, { ...wf, dimension_multiple: 0 }));
  assert.throws(() => app.videoSize(1, { ...wf, max_pixels: 1 }));
});
test("seeds retain integer precision and enforce the signed 64 bit boundary", () => {
  const d = { ...app.videoDefaults, seed: "9007199254740993" };
  assert.equal(app.nextVideoSeed(d, 4), "9007199254740997");
  assert.equal(app.nextVideoSeed({ ...d, seed_mode: "fixed" }, 9), d.seed);
  assert.throws(() => app.nextVideoSeed({ ...d, seed: "9223372036854775807" }));
  const random = BigInt(app.nextVideoSeed({ ...d, seed_mode: "random" }));
  assert.ok(random >= 0n && random <= 9223372036854775807n);
});
test("private flows keep immutable recipes, clips and preferences encrypted and local", async () => {
  login();
  calls.length = 0;
  const repo = app.videoRepository("device"),
    frame = await source();
  const draft = {
    ...app.videoDefaults,
    backend_id: "fixture",
    prompt: "Synthetic prompt",
    first_frame_id: frame,
  };
  await repo.defaults(draft);
  const defaults = await repo.preferences();
  assert.equal(defaults.prompt, "");
  assert.equal(defaults.first_frame_id, null);
  const session = await repo.create(draft),
    id = crypto.randomUUID();
  await repo.submit(session, draft, id);
  draft.prompt = "Changed later";
  const pending = await repo.page(session.id);
  assert.equal(
    JSON.parse(pending.runs[0].draft_json).prompt,
    "Synthetic prompt",
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/api/videos/private");
  assert.equal(calls[0].input.first_frame, png.toString("base64"));
  await repo.finish(session.id, id, { status: "complete", progress: {} });
  assert.deepEqual(
    Buffer.from(await (await repo.result(id)).arrayBuffer()),
    clip,
  );
  assert.equal(calls.filter((v) => v.path.includes("/result")).length, 1);
  await repo.submit(session, JSON.parse(pending.runs[0].draft_json), id);
  assert.equal(calls.filter((v) => v.path === "/api/videos/private").length, 1);
  await assert.rejects(repo.submit(session, draft, id), /already used/);
  assert.equal((await repo.list()).length, 1);
});
test("finishing a download cannot overwrite a newer draft or title", async () => {
  login("concurrency");
  const repo = app.videoRepository("device"),
    draft = { ...app.videoDefaults, first_frame_id: await source() };
  const session = await repo.create(draft),
    id = crypto.randomUUID();
  await repo.submit(session, draft, id);
  let release;
  holdResult = new Promise((resolve) => {
    release = resolve;
  });
  const finish = repo.finish(session.id, id, {
    status: "complete",
    progress: {},
  });
  const current = (await repo.page(session.id)).session;
  await repo.rename(session.id, "Renamed");
  const updated = await repo.save(current, {
    ...draft,
    prompt: "Keep this edit",
  });
  release();
  await finish;
  holdResult = undefined;
  const page = await repo.page(session.id);
  assert.equal(page.session.title, "Renamed");
  assert.equal(page.session.revision, updated.revision);
  assert.equal(JSON.parse(page.session.draft_json).prompt, "Keep this edit");
  assert.equal(page.runs[0].status, "complete");
});
test("repositories cannot carry private data across an account or server switch", async () => {
  login("owner", "first");
  const repo = app.videoRepository("device"),
    session = await repo.create(app.videoDefaults);
  login("other", "first");
  await assert.rejects(repo.page(session.id));
  assert.deepEqual(await app.videoRepository("device").list(), []);
  login("owner", "second");
  await assert.rejects(repo.page(session.id));
  assert.deepEqual(await app.videoRepository("device").list(), []);
});
test("native video export uses the existing document saver and rejects oversized clips", async () => {
  exports.length = 0;
  assert.equal(
    await app.saveFile(new Blob([clip], { type: "video/mp4" }), "clip.mp4"),
    "saved",
  );
  assert.equal(exports[0].name, "native_save_file");
  assert.equal(exports[0].args.mimeType, "video/mp4");
  assert.equal(exports[0].args.data, clip.toString("base64"));
  await assert.rejects(
    app.saveFile(
      new Blob([new Uint8Array(64 * 1024 * 1024 + 1)], { type: "video/mp4" }),
      "big.mp4",
    ),
    /64 MiB/,
  );
});
