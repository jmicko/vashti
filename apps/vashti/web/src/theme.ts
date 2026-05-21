export const THEME_OPTIONS = [
  {
    id: "system",
    label: "System",
    description: "Follow this device's light or dark mode."
  },
  {
    id: "vashti",
    label: "Vashti Green",
    description: "Dark terminal green."
  },
  {
    id: "light",
    label: "Light",
    description: "Bright neutral surfaces with green accents."
  }
] as const;

export type ThemeId = (typeof THEME_OPTIONS)[number]["id"];

const THEME_STORAGE_KEY = "vashti.theme";
const SYSTEM_LIGHT_QUERY = "(prefers-color-scheme: light)";
type AppliedThemeId = Exclude<ThemeId, "system">;

let stopSystemThemeListener: (() => void) | null = null;

export function normalizeTheme(value: string | null | undefined): ThemeId {
  return THEME_OPTIONS.some((option) => option.id === value) ? (value as ThemeId) : "vashti";
}

export function storedTheme(): ThemeId {
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return "vashti";
  }
}

function resolvedTheme(theme: ThemeId): AppliedThemeId {
  if (theme !== "system") {
    return theme;
  }

  return window.matchMedia?.(SYSTEM_LIGHT_QUERY).matches ? "light" : "vashti";
}

function applyResolvedTheme(theme: AppliedThemeId) {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme === "light" ? "light" : "dark";
  const themeColor = theme === "light" ? "#f4f8f0" : "#020402";
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", themeColor);
}

export function applyTheme(theme: ThemeId) {
  stopSystemThemeListener?.();
  stopSystemThemeListener = null;
  applyResolvedTheme(resolvedTheme(theme));

  if (theme !== "system" || !window.matchMedia) {
    return;
  }

  const query = window.matchMedia(SYSTEM_LIGHT_QUERY);
  const syncSystemTheme = () => applyResolvedTheme(resolvedTheme("system"));
  query.addEventListener("change", syncSystemTheme);
  stopSystemThemeListener = () => query.removeEventListener("change", syncSystemTheme);
}

export function storeTheme(theme: ThemeId) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Local storage is a convenience; the signed-in user setting remains authoritative.
  }
}

export function storeAndApplyTheme(theme: ThemeId) {
  storeTheme(theme);
  applyTheme(theme);
}

export function applyStoredTheme() {
  applyTheme(storedTheme());
}
