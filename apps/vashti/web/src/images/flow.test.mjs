import assert from "node:assert/strict";
import test from "node:test";
import { registerHooks } from "node:module";
import { newDraft } from "./types.ts";

// The app uses extensionless TS imports; resolve just this helper's runtime dependency.
const hook = registerHooks({
  resolve(specifier, context, next) {
    if (
      specifier === "./types" &&
      context.parentURL === new URL("./flow.ts", import.meta.url).href
    ) {
      return next(new URL("./types.ts", import.meta.url).href, context);
    }
    return next(specifier, context);
  },
});
const {
  imageTurns,
  imageTreeMessages,
  continuationLeaf,
  hasImageAncestry,
  advancedImageSeed,
} = await import("./flow.ts");
hook.deregister();
const profile = {
  recipe: {
    workflow: "qwen-image-2.1-text-to-image/v1",
    prompt: "synthetic prompt",
    negative_prompt: "",
    seed: "0",
    width: 64,
    height: 64,
    steps: 1,
    cfg: 1,
    attention: "standard",
  },
};
const run = (id, parent = null, retry = null, status = "complete") => ({
  id,
  created_at: 0,
  status,
  request: {
    draft: { ...newDraft(profile), parent_result_id: parent },
    retry_result_id: retry,
    improve_only: false,
  },
});
const result = (
  id,
  run_id,
  group_id = id,
  parent_result_id = null,
  selected = true,
  origin_run_id = run_id,
) => ({
  id,
  run_id,
  origin_run_id,
  group_id,
  parent_result_id,
  selected,
  recipe: profile.recipe,
  created_at: 0,
});

test("new flows use incremental seeds unless the user chose another policy", () => {
  assert.equal(newDraft(profile).seed_mode, "increment");
  assert.equal(
    newDraft({ ...profile, seed_mode: "random" }).seed_mode,
    "random",
  );
});
test("incremental seeds advance after regeneration without overwriting newer settings", () => {
  const generated = run("retry", null, "old");
  generated.completed_count = 1;
  generated.request.draft.recipe.seed = "42";
  const current = newDraft(profile);
  current.recipe.seed = "12";
  assert.equal(advancedImageSeed(generated, 0, current, "12"), "43");
  current.recipe.seed = "99";
  assert.equal(advancedImageSeed(generated, 0, current, "12"), null);
  current.recipe.seed = "42";
  assert.equal(advancedImageSeed(generated, 0, current), "43");
  current.recipe.seed = "43";
  generated.completed_count = 2;
  assert.equal(advancedImageSeed(generated, 1, current), "44");
  current.backend_id = "different-host";
  assert.equal(advancedImageSeed(generated, 1, current), null);
  current.backend_id = null;
  generated.request.draft.seed_mode = "fixed";
  assert.equal(advancedImageSeed(generated, 1, current), null);
  generated.request.draft.seed_mode = "increment";
  generated.request.draft.recipe.seed = "9223372036854775807";
  current.recipe.seed = "9223372036854775807";
  assert.equal(advancedImageSeed(generated, 0, current), null);
});
test("response variants stay under their original prompt with exact children", () => {
  const runs = [run("child", "v1"), run("retry", null, "v1"), run("initial")];
  const results = [
    result("child-image", "child", "child-group", "v1"),
    result("v2", "retry", "group", null, true, "initial"),
    result("v1", "initial", "group", null, false),
  ];
  const oldBranch = imageTurns(runs, results, "child-image");
  assert.deepEqual(
    oldBranch.map((turn) => turn.responses.map((value) => value.id)),
    [["v1"], ["child-image"]],
  );
  const newBranch = imageTurns(runs, results, "v2");
  assert.deepEqual(
    newBranch.map((turn) => turn.responses.map((value) => value.id)),
    [["v2"]],
  );
  assert.equal(continuationLeaf(results, "v1"), "child-image");
  assert.equal(continuationLeaf(results, "v2"), "v2");
  const tree = imageTreeMessages(runs, results, "child-image");
  assert.equal(
    tree.find((value) => value.id === "v2").parent_message_id,
    "prompt:initial",
  );
  assert.equal(
    tree.find((value) => value.id === "prompt:child").parent_message_id,
    "v1",
  );
  assert.equal(
    tree.find((value) => value.id === "prompt:initial").active_child_message_id,
    "v1",
  );
  assert.equal(
    tree.some((value) => value.id === "prompt:retry"),
    false,
  );
});
test("batches keep independent response groups together and failed retries visible", () => {
  const runs = [run("retry", null, "a", "failed"), run("batch")];
  const results = [result("b", "batch"), result("a", "batch")];
  const turns = imageTurns(runs, results, "a");
  assert.equal(turns.length, 1);
  assert.deepEqual(
    turns[0].responses.map((value) => value.id),
    ["a", "b"],
  );
  assert.equal(turns[0].pending.id, "retry");
});
test("pending edits appear only on their exact source branch", () => {
  const results = [
    result("v2", "initial", "group"),
    result("v1", "initial", "group", null, false),
  ];
  const runs = [run("pending", "v1", null, "generating"), run("initial")];
  assert.equal(imageTurns(runs, results, "v1").at(-1).pending.id, "pending");
  assert.equal(
    imageTurns(runs, results, "v2").some((turn) => turn.run.id === "pending"),
    false,
  );
});
test("cycles are bounded and partial histories do not invent a retry prompt", () => {
  const results = [result("v2", "retry", "group", null, true, "initial")];
  const tree = imageTreeMessages([run("retry", null, "old")], results, "v2");
  assert.deepEqual(
    tree.map((value) => value.id),
    ["v2"],
  );
  assert.equal(
    continuationLeaf(
      [result("a", "one", "a", "b"), result("b", "two", "b", "a")],
      "a",
    ),
    "a",
  );
});
test("reopening an older branch loads its exact source prompts and ancestors", () => {
  const history = [
    result("leaf", "edit", "edit-group", "parent"),
    result("parent", "retry", "group", null, true, "original"),
  ];
  assert.equal(hasImageAncestry([run("edit")], history, "leaf"), false);
  assert.equal(
    hasImageAncestry(
      [run("edit"), run("original")],
      history.slice(0, 1),
      "leaf",
    ),
    false,
  );
  assert.equal(
    hasImageAncestry([run("edit"), run("original")], history, "leaf"),
    true,
  );
  assert.equal(hasImageAncestry([], [], null), true);
  assert.equal(hasImageAncestry([], [], "missing"), false);
});
