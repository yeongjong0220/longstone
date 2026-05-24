import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import opentype from "opentype.js"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const fontPath = path.join(__dirname, "fonts/Lobster-Regular.ttf")
const outPath = path.join(__dirname, "../public/splash-source.svg")

const font = opentype.parse(fs.readFileSync(fontPath).buffer)

const W = 1024
const CENTER = W / 2
const TARGET_WIDTH = 720

let fontSize = 200
let textPath = font.getPath("Longstone", 0, 0, fontSize)
let bbox = textPath.getBoundingBox()
fontSize = (fontSize * TARGET_WIDTH) / (bbox.x2 - bbox.x1)

textPath = font.getPath("Longstone", 0, 0, fontSize)
bbox = textPath.getBoundingBox()
const dx = CENTER - (bbox.x1 + bbox.x2) / 2
const dy = CENTER - (bbox.y1 + bbox.y2) / 2

textPath = font.getPath("Longstone", dx, dy, fontSize)
const d = textPath
  .toPathData({ decimalPlaces: 2 })
  .replace(/[A-Z][^A-Z]*NaN[^A-Z]*/g, "")

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${W}" width="${W}" height="${W}">
  <rect width="${W}" height="${W}" fill="#4F46E5"/>
  <path d="${d}" fill="#FFFFFF" transform="matrix(1 0 0 -1 0 ${W})"/>
</svg>
`

fs.writeFileSync(outPath, svg)
const finalBbox = textPath.getBoundingBox()
console.log(`Written ${outPath}`)
console.log(`fontSize=${fontSize.toFixed(2)} bbox=(${finalBbox.x1.toFixed(1)}, ${finalBbox.y1.toFixed(1)})-(${finalBbox.x2.toFixed(1)}, ${finalBbox.y2.toFixed(1)})`)
