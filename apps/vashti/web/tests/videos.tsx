// Isolated UI fixture: every API request is intercepted, never sent to Kuno.
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { VideoWorkspace } from "../src/videos/VideoWorkspace";
import { useVideoAvailability } from "../src/videos/useVideoAvailability";
import { VideoFlowHistory } from "../src/videos/VideoFlowHistory";
import { SettingsPage } from "../src/settingsPage";
import { PwaProvider } from "../src/pwa";
import { storedTheme } from "../src/theme";
import { setPrivateStorageUser } from "../src/privateChatStore";
import {
  videoDefaults,
  nextVideoSeed,
  type VideoDraft,
  type VideoPage,
} from "../src/videos/types";
import type { BackendModelGroup, SettingsSection, User } from "../src/types";
import clipUrl from "../../src/videos/fixtures/clip.mp4?url";
import frameUrl from "../../src/images/fixtures/64.png?url";
import "../src/styles.css";

const fixtureFetch: typeof fetch =
  import.meta.hot?.data.originalFetch ?? window.fetch.bind(window);
if (import.meta.hot) import.meta.hot.data.originalFetch = fixtureFetch;
const clip = await (await fixtureFetch(clipUrl)).blob();
const frame = await (await fixtureFetch(frameUrl)).blob();
const fixtureTheme = storedTheme();
const videos = [
  {
    backend: { id: "fixture", name: "Simulated host" },
    capabilities: {
      api_version: 1,
      access: "inference",
      state: "ready",
      generate_path: "/kuno/v1/videos/generate",
      response_formats: ["application/x-ndjson"],
      workflows: [
        {
          id: videoDefaults.workflow,
          state: "ready",
          first_frame: "required",
          fps: 24,
          frames: [22, 39, 56, 73, 90, 107, 124],
          max_pixels: 452608,
          dimension_multiple: 32,
          min_dimension: 256,
          max_dimension: 1280,
          audio: true,
          output_mime_type: "video/mp4",
        },
      ],
    },
  },
];
const pages = new Map<string, VideoPage>();
const statuses = new Map<string, string>();
let preferences = {
  ...videoDefaults,
  backend_id: "fixture",
  width: 256,
  height: 256,
  frames: 22,
};
let offline = false,
  writes = 0;
let discovery: "ready" | "unsupported" | "offline" = "ready";
const modelGroups: BackendModelGroup[] = [];
const json = (value: unknown) =>
  new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
  });
window.fetch = async (input, options = {}) => {
  const url = new URL(String(input), location.origin),
    path = url.pathname,
    method = options.method ?? "GET";
  const body = typeof options.body === "string" ? JSON.parse(options.body) : {};
  if (path === "/api/videos/hosts") {
    if (discovery === "offline") throw new TypeError("Fixture connection failure");
    if (discovery === "unsupported") return new Response(null, { status: 404 });
    return json(videos);
  }
  if (path === "/api/images/hosts" || path === "/api/images/runs/active")
    return json([]);
  if (path === "/api/images/preferences") return json({});
  if (path === "/api/user-settings") return json({ theme: fixtureTheme });
  if (path === "/api/version") return json({ version: "fixture" });
  if (path === "/api/videos/preferences") {
    if (method === "PUT") {
      preferences = { ...body, prompt: "", first_frame_id: null };
      writes++;
    }
    return json(preferences);
  }
  if (path === "/api/videos/frames" && method === "POST")
    return json({ id: crypto.randomUUID(), width: 64, height: 64 });
  if (path.startsWith("/api/videos/frames/"))
    return new Response(frame, { headers: { "Content-Type": "image/png" } });
  if (path === "/api/videos/sessions") {
    if (method === "POST") {
      const session = {
        id: crypto.randomUUID(),
        title: "New video",
        draft_json: JSON.stringify(body.draft),
        revision: 1,
        updated_at: Date.now() / 1000,
      };
      pages.set(session.id, { session, runs: [] });
      return json(session);
    }
    return json([...pages.values()].map((v) => v.session));
  }
  const session = /^\/api\/videos\/sessions\/([^/]+)(\/(runs|title))?$/.exec(
    path,
  );
  if (session) {
    const page = pages.get(session[1]);
    if (!page) return new Response("Missing", { status: 404 });
    if (session[3] === "title") {
      page.session.title = body.title;
      return json({});
    }
    if (method === "DELETE") {
      pages.delete(session[1]);
      return json({});
    }
    if (method === "PUT") {
      page.session.draft_json = JSON.stringify(body.draft);
      page.session.revision++;
      return json(page.session);
    }
    if (session[3] === "runs") {
      const original = page.runs.find((v) => v.id === body.retry_of);
      const root =
        original && page.runs.find((v) => v.id === original.group_id);
      const draft: VideoDraft = root ? JSON.parse(root.draft_json) : body.draft;
      if (root)
        draft.seed = nextVideoSeed(
          draft,
          page.runs.filter((v) => v.group_id === root.id).length,
        );
      const run = {
        id: body.request_id,
        session_id: session[1],
        group_id: root?.id ?? body.request_id,
        draft_json: JSON.stringify(draft),
        status: "generating",
        error: null,
        size_bytes: 0,
        duration_ms: 0,
        created_at: Date.now() / 1000,
      };
      page.runs.push(run);
      page.session.draft_json = JSON.stringify(draft);
      page.session.title = draft.prompt.slice(0, 60) || "Animated image";
      page.session.revision++;
      statuses.set(run.id, "generating");
      setTimeout(() => {
        if (run.status === "generating") {
          run.status = "complete";
          statuses.set(run.id, "complete");
        }
      }, 2200);
      return json({ id: run.id });
    }
    return json(page);
  }
  if (path === "/api/videos/private") {
    statuses.set(body.request.request_id, "generating");
    setTimeout(() => statuses.set(body.request.request_id, "complete"), 2200);
    return json({ id: body.request.request_id });
  }
  const run = /^\/api\/videos\/runs\/([^/]+)(\/(result|stop))?$/.exec(path);
  if (run) {
    if (offline) return new Response("Offline fixture", { status: 503 });
    if (run[3] === "result")
      return new Response(clip, { headers: { "Content-Type": "video/mp4" } });
    if (run[3] === "stop") {
      statuses.set(run[1], "cancelled");
      for (const page of pages.values())
        for (const value of page.runs)
          if (value.id === run[1]) value.status = "cancelled";
      return json({});
    }
    if (!statuses.has(run[1])) return new Response("Expired", { status: 404 });
    return json({
      status: statuses.get(run[1]),
      progress: { phase: "sampling", step: 4, steps: 8 },
    });
  }
  throw new Error(`Fixture blocked ${method} ${url.origin}${path}`);
};
setPrivateStorageUser(
  "video-fixture",
  { user_id: "video-fixture", key_material: btoa("v".repeat(32)) },
  "video-fixture",
);
function Fixture() {
  const videoAvailable = useVideoAvailability("video-fixture", modelGroups);
  const [id, setId] = useState<string>(),
    [mode, setMode] = useState<"server" | "device">("server"),
    [key, setKey] = useState(0),
    [header, setHeader] = useState<HTMLElement | null>(null),
    [screen, setScreen] = useState("images"),
    [section, setSection] = useState<SettingsSection>("appearance");
  return (
    <PwaProvider>
      <div
        style={{
          height: "100dvh",
          display: "grid",
          gridTemplateRows: "auto auto minmax(0,1fr)",
        }}
      >
        <nav style={{ display: "flex", gap: 8, padding: 8, flexWrap: "wrap" }}>
          <button
            onClick={() => {
              setScreen("images");
            }}
          >
            Video
          </button>
          <button
            onClick={() => {
              setScreen("history");
            }}
          >
            History
          </button>
          <button
            onClick={() => {
              setId(undefined);
              setKey((v) => v + 1);
              setMode((v) => (v === "server" ? "device" : "server"));
              setScreen("images");
            }}
          >
            Switch storage
          </button>
          <button onClick={() => setScreen("settings")}>Settings</button>
          <select aria-label="Fixture video discovery" defaultValue="ready" onChange={(e) => {
            discovery = e.target.value as typeof discovery;
            window.dispatchEvent(new Event("focus"));
          }}>
            <option value="ready">Ready</option>
            <option value="unsupported">Old server</option>
            <option value="offline">Offline</option>
          </select>
          <output aria-label="Video availability">{videoAvailable ? "Available" : "Unavailable"}</output>
          <button
            onClick={() => {
              offline = !offline;
            }}
          >
            Toggle offline
          </button>
        </nav>
        <header
          ref={setHeader}
          className="app-header"
          style={{ minHeight: 60 }}
        />
        {screen === "settings" ? (
          <SettingsPage
            currentUser={
              { id: "fixture", username: "fixture", role: "admin" } as User
            }
            activeSection={section}
            onSelectSection={setSection}
            isAdmin
            modelGroups={[]}
            personas={[]}
            privatePersonas={[]}
            notesStorageMode="server"
            onNotesStorageModeChange={() => {}}
            onSettingsGuardChange={() => {}}
            updateStatus={null}
            updateStatusError={null}
            onUpdateStatusChange={() => {}}
            onRefreshUpdateStatus={async () => null}
            onUserChanged={() => {}}
            onBackendsChanged={async () => {}}
            onToolsChanged={async () => {}}
            onPersonasChanged={async () => {}}
            onPrivatePersonasChanged={async () => {}}
            onContextChanged={async () => {}}
          />
        ) : screen === "history" ? (
          <aside className="chat-history">
            <VideoFlowHistory
              storageMode={mode}
              currentId={id}
              onOpen={(value) => {
                setId(value);
                setScreen("images");
              }}
              onRemoved={() => {
                setId(undefined);
              }}
            />
          </aside>
        ) : (
          <VideoWorkspace
            storageMode={mode}
            sessionId={id}
            newFlowKey={key}
            headerTarget={header}
            onNavigate={(value, nextMode) => {
              setId(value);
              setKey((v) => v + 1);
              if (nextMode) setMode(nextMode);
            }}
          />
        )}
        <output hidden>{writes}</output>
      </div>
    </PwaProvider>
  );
}
createRoot(document.getElementById("root")!).render(<Fixture />);
