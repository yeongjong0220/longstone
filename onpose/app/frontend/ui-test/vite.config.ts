// 격리된 UI 테스트용 Vite 설정.
// 본 앱(:5173)과 완전히 분리: PWA 플러그인 없음, 별도 root(=이 폴더), 별도 포트(:5174).
// node_modules 는 상위 frontend/ 의 것을 그대로 재사용한다(추가 설치 불필요).
import path from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  root: __dirname,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true,
    port: 5174,
    strictPort: true,
  },
})
