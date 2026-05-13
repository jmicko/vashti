export function usesTouchViewport() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse), (max-width: 720px)").matches
  );
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
