export const THEME_OPTIONS = [
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

export function applyTheme(theme: ThemeId) {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme === "light" ? "light" : "dark";
  const themeColor = theme === "light" ? "#f4f8f0" : "#020402";
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", themeColor);
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
