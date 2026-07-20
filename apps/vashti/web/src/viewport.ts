const MOBILE_USER_AGENT_PATTERN =
  /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: {
    mobile?: boolean;
  };
};

function mediaQueryMatches(query: string) {
  return typeof window.matchMedia === "function" && window.matchMedia(query).matches;
}

export function usesMobileInputBehavior() {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return false;
  }

  const navigatorWithUserAgentData = navigator as NavigatorWithUserAgentData;
  if (navigatorWithUserAgentData.userAgentData?.mobile === true) {
    return true;
  }

  if (MOBILE_USER_AGENT_PATTERN.test(navigator.userAgent)) {
    return true;
  }

  const isIPadOS = navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  if (isIPadOS) {
    return true;
  }

  if (mediaQueryMatches("(hover: hover) and (pointer: fine)")) {
    return false;
  }

  if (mediaQueryMatches("(hover: none) and (pointer: coarse)")) {
    return true;
  }

  if (navigatorWithUserAgentData.userAgentData?.mobile === false) {
    return false;
  }

  return window.innerWidth <= 720;
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
  if (!usesMobileInputBehavior()) {
    return;
  }

  const activeElement = document.activeElement;
  if (activeElement instanceof HTMLElement) {
    activeElement.blur();
  }
}
