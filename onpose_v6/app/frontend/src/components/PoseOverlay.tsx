import { useEffect, useRef } from "react"

import type { Landmark } from "@/hooks/usePoseDetector"
import { POSE_CONNECTIONS } from "@/lib/poseConnections"

interface PoseOverlayProps {
  landmarksRef: React.RefObject<Landmark[]>
  /** Mirror horizontally to match front-camera preview. */
  mirror?: boolean
  className?: string
}

const VISIBILITY_THRESHOLD = 0.5

export default function PoseOverlay({
  landmarksRef,
  mirror = true,
  className,
}: PoseOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let raf = 0
    let cancelled = false

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const w = Math.max(1, Math.floor(rect.width * dpr))
      const h = Math.max(1, Math.floor(rect.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
    }

    const draw = () => {
      if (cancelled) return
      resize()
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      const lms = landmarksRef.current
      if (lms && lms.length > 0) {
        const xOf = (lm: Landmark) => (mirror ? 1 - lm[0] : lm[0]) * w
        const yOf = (lm: Landmark) => lm[1] * h

        ctx.strokeStyle = "rgba(110, 231, 183, 0.9)" // emerald-300
        ctx.lineWidth = Math.max(2, w / 360)
        ctx.beginPath()
        for (const [a, b] of POSE_CONNECTIONS) {
          const la = lms[a]
          const lb = lms[b]
          if (!la || !lb) continue
          if (la[3] < VISIBILITY_THRESHOLD || lb[3] < VISIBILITY_THRESHOLD) continue
          ctx.moveTo(xOf(la), yOf(la))
          ctx.lineTo(xOf(lb), yOf(lb))
        }
        ctx.stroke()

        ctx.fillStyle = "rgba(56, 189, 248, 0.95)" // sky-400
        const r = Math.max(3, w / 240)
        for (const lm of lms) {
          if (lm[3] < VISIBILITY_THRESHOLD) continue
          ctx.beginPath()
          ctx.arc(xOf(lm), yOf(lm), r, 0, Math.PI * 2)
          ctx.fill()
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
    }
  }, [landmarksRef, mirror])

  return <canvas ref={canvasRef} className={className} />
}
