import type { AppRoute, NewChatMode, SettingsSection } from "./types";

const settingsSections: SettingsSection[] = [
  "profile",
  "users",
  "backends",
  "models",
  "context",
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

  if (path.startsWith("/app/settings")) {
    const section = path.split("/")[3];
    if (section === "personas") {
      return { page: "settings", section: "models" };
    }
    return { page: "settings", section: isSettingsSection(section) ? section : "profile" };
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
      return { page: "chat", chatId };
    }
  }

  if (path.startsWith("/app/private-chats/")) {
    const chatId = path.split("/")[3];
    if (chatId) {
      return { page: "private-chat", chatId };
    }
  }

  return { page: "chat" };
}

export function pathForRoute(route: AppRoute) {
  if (route.page === "settings") {
    return `/app/settings/${route.section}`;
  }

  if (route.page === "private-chat") {
    return `/app/private-chats/${route.chatId}`;
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
    return `/app/chats/${route.chatId}`;
  }

  return "/app";
}

export function routesEqual(left: AppRoute, right: AppRoute) {
  return pathForRoute(left) === pathForRoute(right);
}
