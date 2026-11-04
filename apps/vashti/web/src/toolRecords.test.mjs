import assert from "node:assert/strict";
import test from "node:test";
import { toolEmptyResult, toolRecordPath, toolRecords } from "./toolRecords.ts";
import { pathForRoute, routeFromLocation } from "./routing.ts";

const usage = (name, result) => ({
  name,
  arguments: {},
  result: JSON.stringify(result),
});
test("memory outcomes link multiple exact records without using model prose", () => {
  const records = toolRecords(
    usage("search_memories", {
      results: [
        { memory_id: "first", excerpt: "First fact" },
        { memory_id: "second", excerpt: "Second fact" },
        { memory_id: "first", excerpt: "First again" },
        null,
        { memory_id: 23 },
      ],
    }),
  );
  assert.deepEqual(
    records.map((value) => value.id),
    ["first", "second"],
  );
  assert.equal(
    toolRecords(
      usage("forget_memory", { memory_id: "forgotten", forgotten: true }),
    )[0].id,
    "forgotten",
  );
  assert.deepEqual(
    toolRecords(usage("remember", { error: "Denied", memory_id: "wrong" })),
    [],
  );
  assert.deepEqual(
    toolRecords({ name: "read_memory", result: "permission denied" }),
    [],
  );
  assert.deepEqual(toolRecords(usage("unknown", { memory_id: "fake" })), []);
});
test("past-chat outcomes deduplicate conversations and preserve titles", () => {
  const record = toolRecords(
    usage("read_chat_history", {
      conversation: { chat_id: "chat", chat_title: "A previous conversation" },
    }),
  )[0];
  assert.equal(record.kind, "chat");
  assert.equal(toolRecordPath(record), "/app/chats/chat");
  assert.equal(
    toolRecords(
      usage("search_chat_history", {
        results: [{ chat_id: "one" }, { chat_id: "one" }],
      }),
    ).length,
    1,
  );
  assert(toolEmptyResult(usage("search_memories", { count: 0, results: [] })));
  assert(!toolEmptyResult(usage("search_notes", { count: 0 })));
});
test("record paths encode IDs and memory deep links round trip", () => {
  const record = { kind: "memory", id: "id &/#?", label: "Memory" };
  const path = toolRecordPath(record);
  const url = new URL(path, "http://localhost");
  globalThis.window = { location: url };
  assert.deepEqual(routeFromLocation(), {
    page: "settings",
    section: "memories",
    memoryId: record.id,
  });
  assert.equal(pathForRoute(routeFromLocation()), path);
  assert.equal(
    toolRecordPath({ kind: "chat", id: "../other", label: "Chat" }),
    "/app/chats/..%2Fother",
  );
});

test("past-chat links show the returned excerpt and bound retained result text", () => {
  const [record] = toolRecords(
    usage("read_chat_history", {
      conversation: {
        chat_id: "chat",
        chat_title: "A conversation",
        target_message_id: "target",
        messages: [
          null,
          { message_id: "other", content: "Other text" },
          { message_id: "target", content: "Selected text" },
        ],
      },
    }),
  );
  assert.equal(record.excerpt, "Selected text");
  const records = toolRecords(
    usage("search_chat_history", {
      results: Array.from({ length: 20 }, (_, index) => ({
        chat_id: String(index),
        chat_title: "x".repeat(1000),
        excerpt: "y".repeat(1000),
      })),
    }),
  );
  assert.equal(records.length, 10);
  assert.equal(records[0].label.length, 120);
  assert.equal(records[0].excerpt.length, 360);
});
