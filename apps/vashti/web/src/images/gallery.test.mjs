import assert from "node:assert/strict";
import test from "node:test";
import { appendGallery, galleryVariants } from "./gallery.ts";

test("gallery pagination appends without replacing or repeating existing tiles", () => {
  const original = [{ id: "one" }, { id: "two" }];
  assert.deepEqual(
    appendGallery(original, [{ id: "two" }, { id: "three" }, { id: "three" }]),
    [...original, { id: "three" }],
  );
  assert.equal(original.length, 2);
});

test("older gallery stacks load across result pages in chronological order", async () => {
  const results = Array.from({ length: 200 }, (_, i) => ({
    id: String(200 - i),
    group_id: i >= 99 && i <= 101 ? "stack" : "other",
  }));
  const pages = [];
  const variants = await galleryVariants(
    { group_id: "stack", variants: 3 },
    async (offset) => {
      pages.push(offset);
      return results.slice(offset, offset + 100);
    },
  );
  assert.deepEqual(pages, [0, 100]);
  assert.deepEqual(
    variants.map((value) => value.id),
    ["99", "100", "101"],
  );
});

test("single images skip result loading and stale counts end at the last page", async () => {
  assert.deepEqual(
    await galleryVariants({ group_id: null, variants: 1 }, () =>
      assert.fail("Must not fetch"),
    ),
    [],
  );
  let requests = 0;
  assert.deepEqual(
    await galleryVariants({ group_id: "missing", variants: 4 }, async () => {
      requests++;
      return [];
    }),
    [],
  );
  assert.equal(requests, 1);
});
