import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "highlight.js/styles/github-dark.css";
import "./styles.css";
import { installAppViewportHeightSync } from "./viewport";

installAppViewportHeightSync();

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
