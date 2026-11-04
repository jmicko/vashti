import assert from "node:assert/strict";
import test from "node:test";
import { applyTheme, normalizeTheme, storedTheme, storeTheme } from "./theme.ts";

test("themes retain legacy preferences and recognize grayscale", () => {
  for (const theme of ["system", "vashti", "grayscale", "light"]) assert.equal(normalizeTheme(theme), theme);
  for (const value of [null, undefined, "unknown"]) assert.equal(normalizeTheme(value), "vashti");
});

test("grayscale uses dark browser chrome and system listeners are replaced cleanly", () => {
  let added = 0, removed = 0, color;
  const root = { dataset: {}, style: {} };
  globalThis.document = { documentElement: root, querySelector: () => ({ setAttribute: (_, value) => { color = value; } }) };
  globalThis.window = { matchMedia: () => ({ matches: true, addEventListener: () => added++, removeEventListener: () => removed++ }) };
  applyTheme("system");
  assert.equal(root.dataset.theme, "light");
  assert.equal(color, "#f7f7f7");
  applyTheme("grayscale");
  assert.equal(root.dataset.theme, "grayscale");
  assert.equal(root.style.colorScheme, "dark");
  assert.equal(color, "#0a0b0a");
  assert.equal(added, 1);
  assert.equal(removed, 1);
});

test("blocked local storage cannot prevent applying a theme", () => {
  globalThis.window = { localStorage: { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } } };
  assert.equal(storedTheme(), "vashti");
  assert.doesNotThrow(() => storeTheme("grayscale"));
});
