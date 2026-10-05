import assert from "node:assert/strict";
import test from "node:test";
import { comparisonReference, ComparisonPress } from "./comparison.ts";

test("comparison uses the first submitted input, not another original or the current draft", () => {
  const result = {
    asset_id: "output",
    input_ids: ["preceding-result", "other-original"],
    original_input_ids: ["start-of-flow"],
  };
  assert.equal(comparisonReference(result), "preceding-result");
  assert.deepEqual(result.input_ids, ["preceding-result", "other-original"]);
  assert.equal(
    comparisonReference({ asset_id: "output", input_ids: [] }),
    undefined,
  );
  assert.equal(
    comparisonReference({ asset_id: "output", input_ids: ["output", "other"] }),
    undefined,
  );
  assert.equal(comparisonReference(), undefined);
});

function control() {
  const changes = [];
  const press = new ComparisonPress((visible) => changes.push(visible));
  return { press, changes };
}

test("pointer release restores the result and competing touches cannot release another hold", () => {
  const { press, changes } = control();
  press.setAvailable(true);
  press.start("pointer:1");
  press.start("pointer:2");
  press.end("pointer:2");
  assert.deepEqual(changes, [true]);
  press.end("pointer:1");
  press.end("pointer:1");
  assert.deepEqual(changes, [true, false]);
});

test("unloaded or unavailable references cannot display an unrelated or blank original", () => {
  const { press, changes } = control();
  press.start("pointer:1");
  press.activate();
  assert.deepEqual(changes, []);
  press.setAvailable(true);
  press.start("pointer:1");
  press.setAvailable(false);
  press.start("pointer:2");
  press.activate();
  assert.deepEqual(changes, [true, false]);
});

test("cancellation, focus loss and variant changes reset holds and ignore stale releases", () => {
  const { press, changes } = control();
  press.setAvailable(true);
  press.start("pointer:1");
  press.reset();
  press.start("pointer:2");
  press.end("pointer:1");
  assert.deepEqual(changes, [true, false, true]);
  press.reset();
  assert.deepEqual(changes, [true, false, true, false]);
});

test("keyboard and assistive activation toggle, while a held pointer retains ownership", () => {
  const { press, changes } = control();
  press.setAvailable(true);
  press.activate();
  press.activate();
  assert.deepEqual(changes, [true, false]);
  press.activate();
  press.start("pointer:1");
  press.activate();
  assert.deepEqual(changes, [true, false, true]);
  press.end("pointer:1");
  assert.deepEqual(changes, [true, false, true, false]);
});
