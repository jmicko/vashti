import assert from "node:assert/strict";
import test from "node:test";
import { appendGallery, galleryDays, galleryVariants } from "./gallery.ts";

test("gallery groups local calendar days without changing library order across pages", () => {
  const now = new Date(2026, 10, 3, 14);
  const item = (id, day, hour = 12) => ({ id, created_at: new Date(2026, 10, day, hour).getTime() / 1000 });
  const items = [item("one", 3), item("two", 3, 0), item("three", 2), item("four", 1)];
  const groups = galleryDays(items, now);
  assert.deepEqual(groups.slice(0, 2).map(day => day.label), ["Today", "Yesterday"]);
  assert.deepEqual(groups.flatMap(day => day.items), items);
  assert.deepEqual(groups.map(day => day.items.length), [2, 1, 1]);
  assert.deepEqual(galleryDays([], now), []);
});

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

test("standalone uploads skip result loading and stale counts end at the last page", async () => {
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

test("single generated images load their submitted reference metadata for comparison", async () => {
  const result = {
    id: "edit",
    asset_id: "output",
    input_ids: ["first-reference", "second-reference"],
    group_id: "single",
  };
  assert.deepEqual(
    await galleryVariants({ group_id: "single", variants: 1 }, async () => [result]),
    [result],
  );
});
