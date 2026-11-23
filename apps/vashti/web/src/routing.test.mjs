import assert from "node:assert/strict";
import test from "node:test";
import { pathForRoute, routeFromLocation, routesEqual } from "./routing.ts";

function readRoute(path) {
  globalThis.window = { location: new URL(path, "http://localhost:5173") };
  return routeFromLocation();
}

test("imported workspace links open Canvas for the correct storage mode", () => {
  for (const page of ["chat","private-chat"]) {
    const route={page,chatId:"imported-page",canvas:true};
    assert.deepEqual(readRoute(pathForRoute(route)),route);
    assert.equal(routesEqual(route,{page,chatId:route.chatId}),false);
  }
});

test("video is its own workspace, not an image view", () => {
  assert.equal(pathForRoute({ page: "videos" }), "/app/videos");
  assert.deepEqual(readRoute("/app/videos"), {
    page: "videos", sessionId: undefined, storageMode: "server",
  });
  assert.equal(readRoute("/app/images").page, "images");
  assert.equal(routesEqual({ page: "images" }, { page: "videos" }), false);
});

test("video links round-trip saved server and device flows", () => {
  for (const storageMode of ["server", "device"]) {
    const route = { page: "videos", sessionId: "a-flow-id", storageMode };
    assert.deepEqual(readRoute(pathForRoute(route)), route);
    assert.equal(pathForRoute(route).includes("mode=device"), storageMode === "device");
  }
});

test("legacy image/video links retain the flow and privacy mode", () => {
  for (const storageMode of ["server", "device"]) {
    const route = readRoute(`/app/images/video#session=old-flow&mode=${storageMode}`);
    assert.deepEqual(route, { page: "videos", sessionId: "old-flow", storageMode });
    assert.ok(pathForRoute(route).startsWith("/app/videos#"));
  }
  assert.equal(readRoute("/app/images/video/").page, "videos");
  assert.equal(readRoute("/app/videos/").page, "videos");
});

test("unknown video storage modes never select device storage", () => {
  assert.equal(readRoute("/app/videos#mode=unknown").storageMode, "server");
});

test("image routes retain their views and independent histories", () => {
  for (const view of ["flow", "edit", "generate", "gallery"]) {
    const route = { page: "images", view, sessionId: "image-flow", storageMode: "device" };
    assert.deepEqual(readRoute(pathForRoute(route)), route);
  }
});

test("custom model editors have independent new, server, and device routes", () => {
  for (const modelEditor of [{}, {personaId:"server-model"}, {personaId:"private-model",storageMode:"device"}]) {
    const route = { page:"settings",section:"models",modelEditor };
    assert.deepEqual(readRoute(pathForRoute(route)), route);
    assert.equal(routesEqual(route, {page:"settings",section:"models"}), false);
  }
});

test("malformed custom model IDs cannot crash initial routing", () => {
  assert.deepEqual(readRoute("/app/settings/models/custom/%invalid"), { page: "settings", section: "models" });
});
