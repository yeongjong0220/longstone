import fs from "node:fs"
import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { VitePWA } from "vite-plugin-pwa"

const certsDir = path.resolve(__dirname, "../backend/certs")
const keyPath = path.join(certsDir, "longstone-key.pem")
const certPath = path.join(certsDir, "longstone-cert.pem")
const hasCerts = fs.existsSync(keyPath) && fs.existsSync(certPath)

// https://vite.dev/config/
export default defineConfig({
  // Vite 8 + 한글 경로 호환 문제 회피: 캐시를 ASCII-only 경로에
  cacheDir: "C:/temp/vite-cache-onpose-v8",
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      devOptions: { enabled: false },  // dev 에서 SW 비활성 — vite 즉시 종료 버그 회피
      includeAssets: ["favicon.svg", "apple-touch-icon-180x180.png"],
      workbox: {
        navigateFallback: "index.html",
        navigateFallbackDenylist: [/^\/api\//, /^\/ws\//],
        globPatterns: ["**/*.{js,css,html,svg,png,ico,woff2}"],
        globIgnores: ["**/mediapipe-wasm/**", "**/models/**"],
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024,
        runtimeCaching: [
          {
            urlPattern: ({ url }) =>
              url.pathname.startsWith("/mediapipe-wasm/") ||
              url.pathname.startsWith("/models/"),
            handler: "CacheFirst",
            options: {
              cacheName: "mediapipe-assets",
              expiration: { maxEntries: 32 },
            },
          },
        ],
      },
      manifest: {
        name: "Longstone",
        short_name: "Longstone",
        description: "온디바이스 AI 자세 교정 코치",
        lang: "ko",
        theme_color: "#3182F6",
        background_color: "#FFFFFF",
        display: "standalone",
        orientation: "any",
        start_url: "/",
        scope: "/",
        icons: [
          { src: "pwa-64x64.png", sizes: "64x64", type: "image/png" },
          { src: "pwa-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512x512.png", sizes: "512x512", type: "image/png" },
          {
            src: "maskable-icon-512x512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      // BlazePose 자동 import 회피 — MoveNet 만 사용. CommonJS export 문제로 vite 가 build 실패.
      "@mediapipe/pose": path.resolve(__dirname, "./src/stubs/mediapipe-pose-stub.ts"),
    },
  },
  optimizeDeps: {
    exclude: ["@mediapipe/pose"],
  },
  server: {
    host: true,
    https: hasCerts
      ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
      : undefined,
  },
  preview: {
    host: true,
    https: hasCerts
      ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
      : undefined,
  },
})
