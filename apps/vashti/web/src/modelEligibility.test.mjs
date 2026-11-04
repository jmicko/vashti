import assert from "node:assert/strict";
import test from "node:test";
import { canChatWith, chatModelGroups } from "./modelEligibility.ts";

test("chat choices exclude known embedding-only models but retain legacy hosts", () => {
  assert.equal(canChatWith({ capabilities: ["embedding"] }), false);
  assert.equal(canChatWith({ capabilities: [" Embedding ", "COMPLETION"] }), true);
  assert.equal(canChatWith({ capabilities: ["completion", "tools"] }), true);
  assert.equal(canChatWith({ capabilities: [] }), true);
  assert.equal(canChatWith({}), true);
});

test("eligible catalogs preserve hosts and do not mutate the full model catalog", () => {
  const groups = [{ backend: { id: "host" }, models: [
    { name: "embed", capabilities: ["embedding"] },
    { name: "chat", capabilities: ["completion"] },
  ] }];
  assert.deepEqual(chatModelGroups(groups)[0].models.map((model) => model.name), ["chat"]);
  assert.equal(groups[0].models.length, 2);
  assert.deepEqual(chatModelGroups([{ ...groups[0], models: [groups[0].models[0]] }])[0].models, []);
});
