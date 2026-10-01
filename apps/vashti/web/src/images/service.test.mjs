import assert from "node:assert/strict";
import test from "node:test";
import { canGenerate, workflowLabel } from "./types.ts";

const recipe = { workflow: "qwen-image-2.1-text-to-image/v1", attention: "standard" };
const generate = { id: recipe.workflow, state: "ready", reference_images_min: 0, reference_images_max: 0, attention: ["standard"] };
const edit = { id: "qwen-image-2.1-edit/v1", state: "ready", reference_images_min: 1, reference_images_max: 10, attention: ["standard", "fast"] };
const service = { state: "ready", capabilities: { images: { workflows: [generate, edit] } } };

test("readiness uses capabilities rather than credential presence", () => {
  assert.equal(canGenerate(service, recipe, 0), true);
  for (const state of ["not_configured", "unreachable", "access_denied", "unsupported", "disabled", "unavailable"]) {
    assert.equal(canGenerate({ ...service, state }, recipe, 0), false);
  }
  assert.equal(canGenerate(undefined, recipe, 0), false);
});

test("partial hosts allow only ready workflows and advertised attention", () => {
  const partial = { state: "partial", capabilities: { images: { workflows: [{ ...generate, state: "unavailable" }, edit] } } };
  assert.equal(canGenerate(partial, recipe, 0), false);
  assert.equal(canGenerate(partial, { ...recipe, workflow: edit.id, attention: "fast" }, 1), true);
  assert.equal(canGenerate(service, { ...recipe, attention: "fast" }, 0), false);
  assert.equal(canGenerate(service, { ...recipe, workflow: "unknown" }, 0), false);
});

test("reference bounds and labels distinguish generation from editing", () => {
  assert.equal(canGenerate(service, recipe, 1), false);
  const editing = { ...recipe, workflow: edit.id };
  assert.equal(canGenerate(service, editing, 0), false);
  assert.equal(canGenerate(service, editing, 1), true);
  assert.equal(canGenerate(service, editing, 10), true);
  assert.equal(canGenerate(service, editing, 11), false);
  assert.equal(workflowLabel(edit.id), "Qwen Image 2.1 Edit");
  assert.equal(workflowLabel("qwen-image-2512-text-to-image/v1"), "Qwen Image 2512");
});
