import assert from "node:assert/strict";
import test from "node:test";
import { AssetCache, thumbnailCrop } from "./assetCache.ts";

const picture = (id) => ({ url: id, width: 320, height: 320, bytes: 100 });

test("cached images share decoding and remain pinned while displayed", async () => {
  let loads = 0;
  const revoked = [];
  const cache = new AssetCache(
    async (id) => {
      loads++;
      return picture(id);
    },
    1,
    1000,
    2,
    (url) => revoked.push(url),
  );
  const first = cache.acquire("first"),
    shared = cache.acquire("first");
  assert.equal(first.promise, shared.promise);
  await first.promise;
  const second = cache.acquire("second");
  await second.promise;
  assert.deepEqual(revoked, []);
  first.release();
  first.release();
  assert.deepEqual(revoked, []);
  shared.release();
  assert.deepEqual(revoked, ["first"]);
  assert.equal(loads, 2);
  assert.equal(cache.peek("second").url, "second");
  second.release();
  cache.dispose();
  assert.deepEqual(revoked, ["first", "second"]);
});

test("image decoding is bounded and queued work is discarded on disposal", async () => {
  const completed = new Map();
  const started = [],
    revoked = [];
  const cache = new AssetCache(
    (id) => {
      started.push(id);
      return new Promise((resolve) =>
        completed.set(id, () => resolve(picture(id))),
      );
    },
    10,
    1000,
    1,
    (url) => revoked.push(url),
  );
  const first = cache.acquire("first"),
    second = cache.acquire("second");
  const rejectedFirst = assert.rejects(first.promise, /closed/);
  const rejectedSecond = assert.rejects(second.promise, /closed/);
  assert.deepEqual(started, ["first"]);
  cache.dispose();
  completed.get("first")();
  await rejectedFirst;
  await rejectedSecond;
  assert.deepEqual(started, ["first"]);
  assert.deepEqual(revoked, ["first"]);
  assert.throws(() => cache.acquire("third"), /closed/);
});

test("decoded byte budget evicts least recently used unpinned images", async () => {
  const revoked = [];
  const cache = new AssetCache(
    async (id) => picture(id),
    10,
    150,
    2,
    (url) => revoked.push(url),
  );
  const first = cache.acquire("first");
  await first.promise;
  first.release();
  const second = cache.acquire("second");
  await second.promise;
  second.release();
  assert.deepEqual(revoked, ["first"]);
  assert.equal(cache.peek("first"), undefined);
  cache.dispose();
});

test("failed image loads can be retried", async () => {
  let attempts = 0;
  const cache = new AssetCache(async (id) => {
    if (!attempts++) throw new Error("Offline");
    return picture(id);
  });
  const failed = cache.acquire("image");
  await assert.rejects(failed.promise, /Offline/);
  failed.release();
  const retried = cache.acquire("image");
  assert.equal((await retried.promise).url, "image");
  retried.release();
  cache.dispose();
});

test("gallery previews crop centrally without changing original dimensions", () => {
  assert.deepEqual(thumbnailCrop(1920, 1080), { x: 420, y: 0, side: 1080 });
  assert.deepEqual(thumbnailCrop(800, 1200), { x: 0, y: 200, side: 800 });
  assert.deepEqual(thumbnailCrop(600, 600), { x: 0, y: 0, side: 600 });
});

test("cached images still validate account identity before reuse", async () => {
  let identity = "first";
  const cache = new AssetCache(
    async (id) => picture(id),
    10,
    1000,
    2,
    () => {},
    () => {
      if (identity !== "first") throw new Error("Identity changed");
    },
  );
  const image = cache.acquire("cached");
  await image.promise;
  image.release();
  identity = "second";
  assert.throws(() => cache.acquire("cached"), /Identity changed/);
  cache.dispose();
});
