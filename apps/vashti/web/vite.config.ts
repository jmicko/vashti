import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

const apiTarget = process.env.VASHTI_DEV_API_URL ?? "http://127.0.0.1:7771";
const apiOrigin = new URL(apiTarget).origin;

export default defineConfig(({ mode }) => {
  const nativeBuild = mode === "native";

  return {
    resolve: nativeBuild
      ? {
          alias: {
            "virtual:pwa-register/react": "/src/pwaRegisterStub.ts"
          }
        }
      : undefined,
    plugins: [
      react(),
      ...(nativeBuild
        ? []
        : [
            VitePWA({
              registerType: "prompt",
              injectRegister: null,
              devOptions: {
                enabled: false
              },
              manifest: {
                id: "/app",
                name: "Vashti",
                short_name: "Vashti",
                description: "A lightweight interface for chatting with Ollama models.",
                start_url: "/app",
                scope: "/",
                display: "standalone",
                background_color: "#020402",
                theme_color: "#020402",
                icons: [
                  {
                    src: "/brand/pwa/vashti-192.png",
                    sizes: "192x192",
                    type: "image/png",
                    purpose: "any"
                  },
                  {
                    src: "/brand/pwa/vashti-512.png",
                    sizes: "512x512",
                    type: "image/png",
                    purpose: "any"
                  },
                  {
                    src: "/brand/pwa/vashti-maskable-512.png",
                    sizes: "512x512",
                    type: "image/png",
                    purpose: "maskable"
                  }
                ]
              },
              workbox: {
                globPatterns: ["**/*.{html,js,css,png,webmanifest}"],
                globIgnores: [
                  "manifest.webmanifest",
                  "brand/pwa/vashti-192.png",
                  "brand/pwa/vashti-512.png",
                  "brand/pwa/vashti-maskable-512.png"
                ],
                navigateFallback: "/index.html",
                navigateFallbackDenylist: [/^\/api(?:\/|$)/],
                cleanupOutdatedCaches: true
              }
            })
          ])
    ],
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
      outDir: nativeBuild ? "dist-native" : "dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 800
    }
  };
});
