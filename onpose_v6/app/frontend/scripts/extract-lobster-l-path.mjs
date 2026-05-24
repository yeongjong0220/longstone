import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import opentype from "opentype.js"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const fontPath = path.join(__dirname, "fonts/Lobster-Regular.ttf")
const outPath = path.join(__dirname, "../public/favicon.svg")

const font = opentype.parse(fs.readFileSync(fontPath).buffer)

const VIEWBOX = 512
const CENTER = VIEWBOX / 2
const TARGET_BBOX = 360

let fontSize = 360
let glyphPath = font.getPath("L", 0, 0, fontSize)
let bbox = glyphPath.getBoundingBox()
const bboxMax = Math.max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
fontSize = (fontSize * TARGET_BBOX) / bboxMax

glyphPath = font.getPath("L", 0, 0, fontSize)
bbox = glyphPath.getBoundingBox()
const dx = CENTER - (bbox.x1 + bbox.x2) / 2
const dy = CENTER - (bbox.y1 + bbox.y2) / 2

glyphPath = font.getPath("L", dx, dy, fontSize)
const d = glyphPath
  .toPathData({ decimalPlaces: 2 })
  .replace(/[A-Z][^A-Z]*NaN[^A-Z]*/g, "")

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" width="${VIEWBOX}" height="${VIEWBOX}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6366F1"/>
      <stop offset="1" stop-color="#4F46E5"/>
    </linearGradient>
  </defs>
  <rect width="${VIEWBOX}" height="${VIEWBOX}" rx="96" fill="url(#bg)"/>
  <path d="${d}" fill="#FFFFFF" transform="matrix(1 0 0 -1 0 ${VIEWBOX})"/>
</svg>
`

fs.writeFileSync(outPath, svg)
const finalBbox = glyphPath.getBoundingBox()
console.log(`Written ${outPath}`)
console.log(`fontSize=${fontSize.toFixed(2)} bbox=(${finalBbox.x1.toFixed(1)}, ${finalBbox.y1.toFixed(1)})-(${finalBbox.x2.toFixed(1)}, ${finalBbox.y2.toFixed(1)})`)
