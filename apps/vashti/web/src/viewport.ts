export function usesTouchViewport() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse), (max-width: 720px)").matches
  );
}

export function updateAppViewportHeight() {
  const nextHeight = Math.round(window.visualViewport?.height ?? window.innerHeight);
  document.documentElement.style.setProperty("--app-height", `${nextHeight}px`);
}

export function installAppViewportHeightSync() {
  updateAppViewportHeight();
  window.addEventListener("resize", updateAppViewportHeight);
  window.addEventListener("orientationchange", updateAppViewportHeight);
  window.visualViewport?.addEventListener("resize", updateAppViewportHeight);
  window.visualViewport?.addEventListener("scroll", updateAppViewportHeight);

  return () => {
    window.removeEventListener("resize", updateAppViewportHeight);
    window.removeEventListener("orientationchange", updateAppViewportHeight);
    window.visualViewport?.removeEventListener("resize", updateAppViewportHeight);
    window.visualViewport?.removeEventListener("scroll", updateAppViewportHeight);
  };
}

export function dismissMobileKeyboard() {
  if (!usesTouchViewport()) {
    return;
  }

  const activeElement = document.activeElement;
  if (activeElement instanceof HTMLElement) {
    activeElement.blur();
  }
}
