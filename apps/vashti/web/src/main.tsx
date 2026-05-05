import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "highlight.js/styles/github-dark.css";
import "./styles.css";

function updateAppViewportHeight() {
  const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
  document.documentElement.style.setProperty("--app-height", `${viewportHeight}px`);
}

updateAppViewportHeight();
window.addEventListener("resize", updateAppViewportHeight);
window.addEventListener("orientationchange", updateAppViewportHeight);
window.visualViewport?.addEventListener("resize", updateAppViewportHeight);
window.visualViewport?.addEventListener("scroll", updateAppViewportHeight);

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
