import assert from "node:assert/strict";
import test from "node:test";
import { aspectRatios, aspectSize, closestAspect } from "./aspectRatio.ts";

test("all presets preserve proportions on the engine grid", () => {
  for (const preset of aspectRatios) {
    const [w, h] = preset.split(":").map(Number);
    const result = aspectSize(w, h, 1024 ** 2);
    for (const dimension of Object.values(result)) {
      assert.equal(dimension % 16, 0);
      assert.ok(dimension >= 64 && dimension <= 2048);
    }
    assert.ok(Math.abs(result.width / result.height / (w / h) - 1) < 0.025, preset);
    assert.ok(Math.abs(result.width * result.height / 1024 ** 2 - 1) < 0.03, preset);
    assert.equal(closestAspect(result.width, result.height), preset);
  }
});

test("original image proportions use the working size, not source pixel count", () => {
  assert.deepEqual(aspectSize(6000, 4000, 1024 ** 2), aspectSize(3, 2, 1024 ** 2));
  assert.deepEqual(aspectSize(4000, 6000, 512 ** 2), aspectSize(2, 3, 512 ** 2));
  assert.deepEqual(aspectSize(1, 1, 512 ** 2), { width: 512, height: 512 });
});

test("extreme inputs stay inside backend dimension limits", () => {
  for (const [w, h, pixels] of [[1, 1000, 1024 ** 2], [1000, 1, 1024 ** 2], [21, 9, 2048 ** 2], [1, 1, 1]]) {
    for (const n of Object.values(aspectSize(w, h, pixels))) assert.ok(n >= 64 && n <= 2048 && n % 16 === 0);
  }
  for (const bad of [0, -1, NaN, Infinity]) assert.throws(() => aspectSize(bad, 1, 1024 ** 2));
  assert.equal(closestAspect(1200, 1104), "Custom");
});
