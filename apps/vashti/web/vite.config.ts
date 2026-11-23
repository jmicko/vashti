import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { createRequire } from "node:module";
import { pdfAssets } from "./pdfAssets";

const apiTarget = process.env.VASHTI_DEV_API_URL ?? "http://127.0.0.1:7771";
const apiOrigin = new URL(apiTarget).origin;
const resolvePackage = createRequire(import.meta.url).resolve;

export default defineConfig(({ mode }) => {
  const nativeBuild = mode === "native";

  return {
    optimizeDeps: {
      // Prebundle the lazy Canvas and ZIP worker before the app loads in dev.
      include: [
        "@codemirror/commands", "@codemirror/language", "@codemirror/merge",
        "@codemirror/search", "@codemirror/state", "@codemirror/view",
        "@codemirror/lang-css", "@codemirror/lang-html", "@codemirror/lang-javascript",
        "@codemirror/lang-json", "@codemirror/lang-markdown", "@codemirror/lang-python",
        "@codemirror/lang-rust", "@codemirror/lang-sql",
        "@zip.js/zip.js/lib/zip-core-native.js",
        "pdfjs-dist/legacy/build/pdf.mjs", "yjs", "tesseract.js"
      ]
    },
    resolve: {
      alias: {
        // Markdown also runs in the ZIP worker, where a DOM decoder cannot run.
        "decode-named-character-reference": resolvePackage("decode-named-character-reference"),
        ...(nativeBuild ? { "virtual:pwa-register/react": "/src/pwaRegisterStub.ts" } : {})
      }
    },
    plugins: [
      react(),
      pdfAssets(),
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
                  "ocr-runtime/**",
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
