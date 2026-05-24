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
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      devOptions: { enabled: true },
      includeAssets: ["favicon.svg", "apple-touch-icon-180x180.png", "apple-splash-*.png"],
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
        theme_color: "#4F46E5",
        background_color: "#E8E4F8",
        display: "standalone",
        orientation: "portrait",
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
    },
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
