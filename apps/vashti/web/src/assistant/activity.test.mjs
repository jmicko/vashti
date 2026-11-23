import assert from "node:assert/strict";
import test from "node:test";
import { actionCursor, activityEntries, mergeActivity } from "./activity.ts";
import { pathForRoute, routeFromLocation } from "../routing.ts";

test("action cursor never skips an unfinished or pending action", () => {
  const completed = { id: "a", ordinal: 0, status: "completed", finished_at: 3 };
  assert.equal(actionCursor([completed]), 0);
  assert.equal(actionCursor([completed, { ordinal: 1, status: "rejected", finished_at: null }]), 0);
  assert.equal(actionCursor([completed, { ordinal: 1, status: "pending" }, { ordinal: 2, status: "completed", finished_at: 4 }]), 0);
  assert.equal(actionCursor([{ ordinal: 1, status: "completed", finished_at: 4 }]), -1);
});
test("incremental activity preserves completed records and their component identity", () => {
  const done = { id: "a", ordinal: 0, status: "completed", result_text: "saved", finished_at: 3 };
  const pending = { id: "b", ordinal: 1, status: "pending" };
  const previous = { run: { id: "run" }, events: [{ id: 1 }], actions: [done, pending], artifacts: [] };
  const next = { run: { id: "run" }, events: [{ id: 1 }, { id: 2 }], actions: [{ ...pending, status: "completed", result_text: "new" }], artifacts: [] };
  const merged = mergeActivity(previous, next);
  assert.equal(merged.actions[0], done);
  assert.equal(merged.actions[1].result_text, "new");
  assert.deepEqual(merged.events.map(e => e.id), [1, 2]);
  assert.deepEqual(mergeActivity(merged, { ...next, run: { id: "other" } }).actions, next.actions);
});
test("Assistant routes preserve exact task and run selection", () => {
  for (const route of [{ page: "assistant" }, ...["new", "defaults", "tasks", "conversations", "compose", "notifications"].map(view => ({ page: "assistant", view })), { page: "assistant", conversationId: "conversation id" }, { page: "assistant", taskId: "id", runId: "run" }, { page: "assistant", runId: "run" }]) {
    const url = new URL(pathForRoute(route), "https://fixture.invalid");
    globalThis.window = { location: url };
    assert.equal(pathForRoute(routeFromLocation()), pathForRoute(route));
  }
});
test("same-second tool actions remain before the response that depends on them", () => {
  const events = [{ id: 1, kind: "user", created_at: 100 }, { id: 2, kind: "action", created_at: 100, data_json: '{"action_id":"calc"}' }, { id: 3, kind: "assistant", created_at: 100 }];
  const actions = [{ id: "calc", requested_at: 100, ordinal: 0 }];
  assert.deepEqual(activityEntries(events, actions).map(e => e.key), ["e1", "acalc", "e3"]);
});
