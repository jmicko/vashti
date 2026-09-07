import assert from "node:assert/strict";
import test from "node:test";
import {
  modelInfoForValue,
  modelValue,
  personaModelValue,
  privatePersonaModelValue,
  selectedModelBaseParts
} from "./modelSelection.ts";

const groups = [
  { backend: { id: "original" }, models: [{ name: "original:model", supports_images: true }] },
  { backend: { id: "replacement" }, models: [{ name: "replacement:model", supports_images: false }] }
];
const version = {
  id: "version",
  persona_id: "custom",
  base_backend_id: "original",
  base_model_name: "original:model",
  system_prompt: "Keep the custom prompt",
  background_asset_id: "custom-background"
};
const persona = { id: "custom", current_version: version };
const replacement = modelValue("replacement", "replacement:model");

for (const device of [false, true]) {
  test(`${device ? "device" : "server"} custom model overrides only its inference base`, () => {
    const value = device ? privatePersonaModelValue(version.id) : personaModelValue(version.id);
    const personas = device ? [] : [persona];
    const privatePersonas = device ? [persona] : [];
    const args = [groups, personas, privatePersonas, value, [], []];
    assert.deepEqual(selectedModelBaseParts(...args, replacement), {
      backendId: "replacement", modelName: "replacement:model"
    });
    const info = modelInfoForValue(...args, replacement);
    assert.equal(info.name, "replacement:model");
    assert.equal(info.supports_images, false);
    assert.equal(info.background_asset_id, "custom-background");
    assert.equal(info.background_is_private, device);
    assert.deepEqual(selectedModelBaseParts(...args, null), {
      backendId: "original", modelName: "original:model"
    });
    assert.equal(version.base_model_name, "original:model");
    assert.equal(version.system_prompt, "Keep the custom prompt");
  });
}

test("conversation overrides do not change a regular model selection", () => {
  const args = [groups, [], [], modelValue("original", "original:model"), [], []];
  assert.deepEqual(selectedModelBaseParts(...args, replacement), {
    backendId: "original", modelName: "original:model"
  });
  assert.equal(modelInfoForValue(...args, replacement).name, "original:model");
});

test("historical custom versions can use a different base without changing the template", () => {
  const current = { ...version, id: "newer-version", base_model_name: "newer:model" };
  const args = [groups, [{ ...persona, current_version: current }], [], personaModelValue(version.id), [version], []];
  assert.equal(selectedModelBaseParts(...args, replacement).modelName, "replacement:model");
  assert.equal(selectedModelBaseParts(...args, null).modelName, "original:model");
  assert.equal(current.base_model_name, "newer:model");
});
