import assert from "node:assert/strict";
import test from "node:test";
import { noteAccessStatus, memoryAccessStatus } from "./effectiveAccess.ts";
const settings = {
  allow_model_read: true,
  allow_model_edit: true,
  allow_model_trash: true,
  allow_model_forget: true,
};
const scope = { all_models: false, model_keys: ["persona:one"] };
test("effective note access intersects account, note, and exact model restrictions", () => {
  const note = { ai_access: "manage", model_scope: scope };
  assert(
    noteAccessStatus(note, settings, "persona:one").every(
      (value) => value.allowed,
    ),
  );
  assert(
    noteAccessStatus(note, settings, "persona:other").every(
      (value) => !value.allowed,
    ),
  );
  assert(
    noteAccessStatus(note, { ...settings, allow_model_read: false }).every(
      (value) => !value.allowed,
    ),
  );
  assert.deepEqual(
    noteAccessStatus({ ...note, ai_access: "read" }, settings).map(
      (value) => value.allowed,
    ),
    [true, false, false],
  );
  assert(
    noteAccessStatus(note, settings, "", true).every(
      (value) => !value.allowed && value.reason === "In Trash",
    ),
  );
  assert(
    !noteAccessStatus(
      { ...note, model_scope: { all_models: false, model_keys: [] } },
      settings,
    )[0].allowed,
  );
});
test("forgotten and unscoped memories remain unavailable even with account access enabled", () => {
  assert(
    memoryAccessStatus(scope, settings, "persona:one").every(
      (value) => value.allowed,
    ),
  );
  assert(
    memoryAccessStatus(scope, settings, "persona:one", true).every(
      (value) => !value.allowed,
    ),
  );
  assert(
    memoryAccessStatus(scope, settings, "persona:two").every(
      (value) => !value.allowed,
    ),
  );
  assert(
    memoryAccessStatus(scope, { ...settings, allow_model_read: false }).every(
      (value) => !value.allowed,
    ),
  );
  assert.deepEqual(
    memoryAccessStatus(scope, { ...settings, allow_model_edit: false }).map(
      (value) => value.allowed,
    ),
    [true, false, true],
  );
});
