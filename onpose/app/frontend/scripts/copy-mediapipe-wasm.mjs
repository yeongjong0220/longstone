// node_modules/@mediapipe/tasks-vision/wasm → public/mediapipe-wasm/
// 오프라인 동작을 위해 WASM을 로컬에서 호스팅. gitignore되어 있으니 매 install마다 복사.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const src = resolve(here, "../node_modules/@mediapipe/tasks-vision/wasm")
const dst = resolve(here, "../public/mediapipe-wasm")

if (!existsSync(src)) {
  console.warn(`[copy-mediapipe-wasm] skip: ${src} not found`)
  process.exit(0)
}

mkdirSync(dst, { recursive: true })
for (const f of readdirSync(src)) {
  copyFileSync(resolve(src, f), resolve(dst, f))
}
console.log(`[copy-mediapipe-wasm] ok: ${readdirSync(dst).length} files → public/mediapipe-wasm/`)
