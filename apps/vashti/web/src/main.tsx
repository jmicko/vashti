import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import { PwaProvider } from "./pwa";
import { markPerformance } from "./performance";
import { applyStoredTheme } from "./theme";
import { installAppViewportHeightSync } from "./viewport";
import { NativeConnectionProvider } from "./nativeConnections";

markPerformance("vashti:app-start");
applyStoredTheme();
installAppViewportHeightSync();

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <NativeConnectionProvider>
      <PwaProvider>
        <App />
      </PwaProvider>
    </NativeConnectionProvider>
  </React.StrictMode>
);
