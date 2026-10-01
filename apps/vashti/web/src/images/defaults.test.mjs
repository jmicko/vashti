import assert from "node:assert/strict";
import test from "node:test";
import { newDraft, newFlowWithSettings, profileFromDraft, resetDraftSettings } from "./types.ts";

const profile = {
  recipe: { workflow: "qwen-image-2512-text-to-image/v1", prompt: "", negative_prompt: "blur",
    width: 1024, height: 768, steps: 30, cfg: 4, flow_shift: 4.2, seed: "15", attention: "standard" },
  model_may_change: ["size"], backend_id: "saved-host", seed_mode: "random", count: 3,
  auto_improve: true, prompt_help: { backend_id: "helper-host", model_name: "helper", guidance: "Add detail" },
};
const changed = () => ({ ...newDraft(profile), recipe: { ...profile.recipe, workflow: "qwen-image-2.1-text-to-image/v1",
  prompt: "A teapot", negative_prompt: "text", steps: 40, flow_shift: null }, backend_id: "other-host",
  input_ids: ["source"], original_input_ids: ["original"], parent_result_id: "parent", prompt_version_id: "prompt-version",
  count: 5, seed_mode: "fixed", auto_improve: false });

test("new flows use account defaults, not edits in another flow", () => {
  const first = newDraft(profile);
  first.recipe.steps = 99;
  first.prompt_help.guidance = "Changed in this flow";
  const next = newDraft(profile);
  assert.equal(next.recipe.steps, 30);
  assert.equal(next.prompt_help.guidance, "Add detail");
  assert.equal(next.backend_id, "saved-host");
  assert.equal(next.count, 3);
  assert.equal(next.auto_improve, true);
  assert.equal(newDraft({ recipe: profile.recipe, model_may_change: [] }).count, 1);
  assert.equal(newDraft({ ...profile, count: null }).count, null);
});

test("explicit save captures every setting without copying content or changing permissions", () => {
  const draft = changed();
  const saved = profileFromDraft(profile, draft, "original");
  assert.equal(saved.recipe.prompt, "");
  assert.equal(saved.recipe.workflow, draft.recipe.workflow);
  assert.equal(saved.backend_id, "other-host");
  assert.equal(saved.count, 5);
  assert.equal(saved.seed_mode, "fixed");
  assert.equal(saved.auto_improve, false);
  assert.equal(saved.edit_mode, "original");
  assert.deepEqual(saved.model_may_change, ["size"]);
  saved.recipe.steps = 1;
  saved.prompt_help.guidance = "New defaults";
  assert.equal(draft.recipe.steps, 40);
  assert.equal(draft.prompt_help.guidance, "Add detail");
  assert.equal(profile.recipe.steps, 30);
});

test("reset restores the default model and all settings while retaining the current prompt", () => {
  const draft = changed();
  const reset = resetDraftSettings(draft, profile);
  assert.deepEqual(reset.recipe, { ...profile.recipe, prompt: "A teapot" });
  assert.equal(reset.backend_id, "saved-host");
  assert.equal(reset.count, 3);
  assert.equal(reset.seed_mode, "random");
  assert.equal(reset.auto_improve, true);
  assert.equal(reset.parent_result_id, "parent");
  assert.equal(reset.prompt_version_id, "prompt-version");
  assert.deepEqual(reset.input_ids, []);
  const editProfile = { ...profile, recipe: { ...profile.recipe, workflow: "qwen-image-2.1-edit/v1" } };
  assert.deepEqual(resetDraftSettings(draft, editProfile).input_ids, ["source"]);
  assert.deepEqual(resetDraftSettings(draft, editProfile).original_input_ids, ["original"]);
});

test("new flow with current settings has independent settings and no inherited content or branches", () => {
  const draft = changed();
  const next = newFlowWithSettings(draft);
  assert.equal(next.recipe.prompt, "");
  assert.equal(next.recipe.steps, 40);
  assert.equal(next.backend_id, "other-host");
  assert.equal(next.count, 5);
  assert.equal(next.seed_mode, "fixed");
  assert.equal(next.parent_result_id, null);
  assert.equal(next.prompt_version_id, null);
  assert.deepEqual(next.input_ids, []);
  assert.deepEqual(next.original_input_ids, []);
  next.prompt_help.guidance = "Different helper settings";
  assert.equal(draft.prompt_help.guidance, "Add detail");
  assert.equal(profile.recipe.steps, 30);
});
