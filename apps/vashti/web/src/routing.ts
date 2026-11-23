import type { AppRoute, NewChatMode, SettingsSection } from "./types";

const settingsSections: SettingsSection[] = [
  "profile",
  "appearance",
  "device",
  "users",
  "backends",
  "models",
  "context",
  "notes",
  "memories",
  "tools",
  "app"
];

const newChatModeStorageKey = "vashti:new-chat-mode";

function isSettingsSection(value: string | undefined): value is SettingsSection {
  return settingsSections.includes(value as SettingsSection);
}

export function storedNewChatMode(): NewChatMode {
  try {
    return window.localStorage.getItem(newChatModeStorageKey) === "private"
      ? "private"
      : "standard";
  } catch {
    return "standard";
  }
}

export function storeNewChatMode(mode: NewChatMode) {
  try {
    window.localStorage.setItem(newChatModeStorageKey, mode);
  } catch {
    // The toggle still works for this session if browser storage is unavailable.
  }
}

export function routeFromLocation(): AppRoute {
  const path = window.location.pathname;

  if (path === "/app/assistant" || path.startsWith("/app/assistant/")) {
    const [, , , section, id] = path.split("/");
    if (section === "tasks" && id) return { page: "assistant", taskId: decodeURIComponent(id), runId: new URLSearchParams(window.location.hash.slice(1)).get("run") ?? undefined };
    if (section === "runs" && id) return { page: "assistant", runId: decodeURIComponent(id) };
    if (section === "conversations" && id) return { page: "assistant", conversationId: decodeURIComponent(id) };
    return { page: "assistant", view: section === "new" || section === "defaults" || section === "tasks" || section === "conversations" || section === "compose" || section === "notifications" ? section : undefined };
  }

  if (path === "/app/videos" || path.startsWith("/app/videos/") || path === "/app/images/video" || path === "/app/images/video/") {
    const params = new URLSearchParams(window.location.hash.slice(1));
    return { page: "videos", sessionId: params.get("session") ?? undefined, storageMode: params.get("mode") === "device" ? "device" : "server" };
  }

  if (path === "/app/images" || path.startsWith("/app/images/")) {
    const view = path.split("/")[3];
    const params = new URLSearchParams(window.location.hash.slice(1));
    const sessionId = params.get("session") ?? undefined;
    return { page: "images", view: view === "generate" || view === "edit" || view === "gallery" || view === "flow" ? view : "home", sessionId, storageMode: params.get("mode") === "device" ? "device" : "server" };
  }

  if (path.startsWith("/app/settings")) {
    const section = path.split("/")[3];
    if (section === "personas") {
      return { page: "settings", section: "models" };
    }
    const selectedSection = isSettingsSection(section) ? section : "profile";
    if (selectedSection === "models" && path.split("/")[4] === "custom" && path.split("/")[5]) {
      const id = path.split("/")[5];
      const mode = new URLSearchParams(window.location.hash.slice(1)).get("mode");
      let personaId: string | undefined;
      try {
        personaId = id === "new" ? undefined : decodeURIComponent(id);
      } catch {
        return { page: "settings", section: "models" };
      }
      return { page: "settings", section: "models", modelEditor: {
        ...(personaId ? { personaId } : {}),
        ...(mode === "device" ? { storageMode: "device" } : {})
      } };
    }
    if (selectedSection === "memories") {
      const memoryId = new URLSearchParams(window.location.hash.slice(1)).get("memory")?.trim();
      return { page: "settings", section: "memories", ...(memoryId ? { memoryId } : {}) };
    }
    if (selectedSection === "notes") {
      const storageMode = new URLSearchParams(window.location.hash.slice(1)).get("mode");
      return storageMode === "device"
        ? { page: "settings", section: "notes", storageMode: "device" }
        : { page: "settings", section: "notes", storageMode: "server" };
    }
    return { page: "settings", section: selectedSection };
  }

  if (path === "/app/notes" || path.startsWith("/app/notes/")) {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const storageMode = params.get("mode");
    const noteId = params.get("note")?.trim();
    if (storageMode === "server" || storageMode === "device") {
      return noteId
        ? { page: "notes", storageMode, noteId }
        : { page: "notes", storageMode };
    }
    return { page: "notes" };
  }

  if (path.startsWith("/app/chats/")) {
    const chatId = path.split("/")[3];
    if (chatId) {
      return { page: "chat", chatId, ...(window.location.hash==="#canvas"?{canvas:true}:{}) };
    }
  }

  if (path.startsWith("/app/private-chats/")) {
    const chatId = path.split("/")[3];
    if (chatId) {
      return { page: "private-chat", chatId, ...(window.location.hash==="#canvas"?{canvas:true}:{}) };
    }
  }

  return { page: "chat" };
}

export function pathForRoute(route: AppRoute) {
  if (route.page === "assistant") {
    if (route.conversationId) return `/app/assistant/conversations/${encodeURIComponent(route.conversationId)}`;
    if (route.taskId) return `/app/assistant/tasks/${encodeURIComponent(route.taskId)}${route.runId ? `#${new URLSearchParams({ run: route.runId })}` : ""}`;
    if (route.runId) return `/app/assistant/runs/${encodeURIComponent(route.runId)}`;
    return `/app/assistant${route.view ? `/${route.view}` : ""}`;
  }
  if (route.page === "images" || route.page === "videos") {
    const params = new URLSearchParams();
    if (route.storageMode === "device") params.set("mode", "device");
    if (route.sessionId) params.set("session", route.sessionId);
    const path = route.page === "videos"
      ? "/app/videos"
      : `/app/images${!route.view || route.view === "home" ? "" : `/${route.view}`}`;
    return `${path}${params.size ? `#${params}` : ""}`;
  }
  if (route.page === "settings") {
    if (route.section === "models" && route.modelEditor) {
      return `/app/settings/models/custom/${route.modelEditor.personaId ? encodeURIComponent(route.modelEditor.personaId) : "new"}${route.modelEditor.storageMode === "device" ? "#mode=device" : ""}`;
    }
    if (route.section === "memories" && route.memoryId) {
      return `/app/settings/memories#${new URLSearchParams({ memory: route.memoryId })}`;
    }
    if (route.section === "notes" && route.storageMode === "device") {
      return "/app/settings/notes#mode=device";
    }
    return `/app/settings/${route.section}`;
  }

  if (route.page === "private-chat") {
    return `/app/private-chats/${route.chatId}${route.canvas?"#canvas":""}`;
  }

  if (route.page === "notes") {
    if (!route.storageMode) {
      return "/app/notes";
    }
    const params = new URLSearchParams({ mode: route.storageMode });
    if (route.noteId) {
      params.set("note", route.noteId);
    }
    return `/app/notes#${params.toString()}`;
  }

  if (route.chatId) {
    return `/app/chats/${route.chatId}${route.canvas?"#canvas":""}`;
  }

  return "/app";
}

export function routesEqual(left: AppRoute, right: AppRoute) {
  return pathForRoute(left) === pathForRoute(right);
}
