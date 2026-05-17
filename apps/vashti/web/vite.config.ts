import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VASHTI_DEV_API_URL ?? "http://127.0.0.1:7771";
const apiOrigin = new URL(apiTarget).origin;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("Origin", apiOrigin);
          });
        }
      }
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 800
  }
});
