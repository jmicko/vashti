import assert from "node:assert/strict";
import test from "node:test";
import { boundPhoto, fitPhoto, fittedView, photoSwipe, zoomPhoto } from "./photoGeometry.ts";

test("fit contains different orientations without cropping or stretching", () => {
  assert.deepEqual(fitPhoto({ width: 1200, height: 800 }, { width: 390, height: 844 }), { width: 390, height: 260 });
  assert.deepEqual(fitPhoto({ width: 800, height: 1200 }, { width: 844, height: 390 }), { width: 260, height: 390 });
});
test("panning remains bounded on each image axis and fit always recenters", () => {
  const image = { width: 800, height: 800 }, viewport = { width: 400, height: 800 };
  assert.deepEqual(boundPhoto({ scale: 2, x: 900, y: 900 }, image, viewport), { scale: 2, x: 200, y: 0 });
  assert.deepEqual(boundPhoto({ scale: 0.2, x: 900, y: -900 }, image, viewport), fittedView);
  assert.equal(boundPhoto({ scale: Infinity, x: 0, y: 0 }, image, viewport).scale, 1);
  assert.equal(boundPhoto({ scale: 99, x: 0, y: 0 }, image, viewport).scale, 8);
});
test("zoom anchors the detail beneath the finger and supports a moving pinch midpoint", () => {
  const view = { scale: 2, x: -30, y: 20 };
  const next = zoomPhoto(view, 4, [100, 50]);
  assert.equal((100 - next.x) / next.scale, (100 - view.x) / view.scale);
  assert.equal((50 - next.y) / next.scale, (50 - view.y) / view.scale);
  assert.deepEqual(zoomPhoto(view, 2, [100, 50], [120, 70]), { scale: 2, x: -10, y: 40 });
});
test("incidental movement does not fling and one intentional swipe advances one photo", () => {
  assert.equal(photoSwipe(4, 9, 390), 0);
  assert.equal(photoSwipe(30, 0.2, 390), 0);
  assert.equal(photoSwipe(-120, 0.2, 390), 1);
  assert.equal(photoSwipe(35, 0.8, 390), -1);
  assert.equal(photoSwipe(-900, 4, 390), 1);
});
