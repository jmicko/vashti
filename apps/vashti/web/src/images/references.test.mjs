import assert from "node:assert/strict";
import test from "node:test";
import {
  referencePatch,
  referenceWorkflow,
  toggleReference,
} from "./references.ts";

const draft = {
  recipe: {
    workflow: "generate",
    prompt: "Keep this prompt",
    negative_prompt: "Keep this too",
    steps: 20,
  },
  input_ids: [],
  original_input_ids: [],
  parent_result_id: "previous",
};
const profile = {
  recipe: { workflow: "edit", steps: 8, prompt: "", negative_prompt: "" },
};
const edit = { id: "edit", reference_images_max: 10 };

test("gallery references retain selection order and do not duplicate existing references", () => {
  const current = {
    ...draft,
    input_ids: ["existing"],
    recipe: { ...draft.recipe, workflow: "edit" },
  };
  const patch = referencePatch(
    current,
    ["second", "existing", "third"],
    profile,
    edit,
  );
  assert.deepEqual(patch.input_ids, ["existing", "second", "third"]);
  assert.deepEqual(patch.original_input_ids, patch.input_ids);
  assert.equal(patch.parent_result_id, "previous");
  assert.equal(patch.recipe, undefined);
  assert.deepEqual(current.input_ids, ["existing"]);
});

test("attaching the first gallery or uploaded reference chooses an edit workflow without losing the prompt", () => {
  const patch = referencePatch(draft, ["first"], profile, edit);
  assert.deepEqual(patch.recipe, {
    ...profile.recipe,
    prompt: draft.recipe.prompt,
    negative_prompt: draft.recipe.negative_prompt,
  });
  assert.equal(patch.parent_result_id, null);
  assert.deepEqual(patch.input_ids, ["first"]);
  assert.equal(draft.recipe.workflow, "generate");
});

test("reference limits include existing inputs and respect the inference model limit", () => {
  const current = {
    ...draft,
    input_ids: Array.from({ length: 10 }, (_, i) => String(i)),
  };
  assert.throws(
    () => referencePatch(current, ["extra"], profile, edit),
    /at most 10/,
  );
  assert.throws(
    () =>
      referencePatch(draft, ["one", "two"], profile, {
        ...edit,
        reference_images_max: 1,
      }),
    /at most 1/,
  );
  assert.doesNotThrow(() => referencePatch(current, ["0"], profile, edit));
});

test("gallery selection can be removed at the limit and reselected in a new order", () => {
  const ids = ["one", "two"];
  assert.equal(toggleReference(ids, "three", 2), ids);
  assert.deepEqual(toggleReference(ids, "one", 2), ["two"]);
  assert.deepEqual(toggleReference(["two"], "one", 2), ["two", "one"]);
  assert.deepEqual(toggleReference([], "one", 0), []);
});

test("attachments preserve the selected ready edit model rather than choosing the first model", () => {
  const workflows = [
    { id: "generate", reference_images_max: 0, state: "ready" },
    { id: "offline", reference_images_max: 10, state: "unavailable" },
    { id: "first-edit", reference_images_max: 1, state: "ready" },
    { id: "chosen-edit", reference_images_max: 10, state: "ready" },
  ];
  assert.equal(referenceWorkflow(workflows, "chosen-edit").id, "chosen-edit");
  assert.equal(referenceWorkflow(workflows, "generate").id, "first-edit");
  assert.equal(referenceWorkflow(workflows.slice(0, 2), "offline"), undefined);
});

test("references added while hosts are unavailable retain the draft until a model can be selected", () => {
  const patch = referencePatch(draft, ["source"], profile);
  assert.equal(patch.recipe, undefined);
  assert.equal(patch.parent_result_id, null);
  assert.deepEqual(patch.input_ids, ["source"]);
});

test("adding references does not rewrite an existing ordered input list", () => {
  const current = { ...draft, input_ids: ["one", "one"] };
  assert.deepEqual(referencePatch(current, ["two", "two"], profile).input_ids, [
    "one",
    "one",
    "two",
  ]);
});
