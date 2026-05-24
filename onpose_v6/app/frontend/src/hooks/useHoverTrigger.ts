import { useEffect, useRef, useState } from "react"

import type { Landmark } from "@/hooks/usePoseDetector"

/** MediaPipe Pose: 15=LEFT_WRIST, 16=RIGHT_WRIST */
const LEFT_WRIST = 15
const RIGHT_WRIST = 16
const VIS_THR = 0.5

export type HoverTarget = {
  /** Unique key (must be stable). Used as the selection result. */
  key: string
  /** Target rect in viewport coords (use getBoundingClientRect()). */
  rect: DOMRect
}

type UseHoverTriggerOptions = {
  landmarksRef: React.RefObject<Landmark[]>
  /** Mirror x to match front-camera preview ([transform:scaleX(-1)] on the <video>). */
  mirror?: boolean
  /** Time (ms) the cursor must stay inside a target before it is selected. */
  dwellMs?: number
  /** Called once when dwell completes. */
  onSelect: (key: string) => void
  /** When false, the loop is paused. */
  enabled?: boolean
  /**
   * Bounds of the visible camera area (must already be on screen).
   * Used to map normalized MediaPipe coords → screen pixels.
   */
  cameraBoundsRef: React.RefObject<DOMRect | null>
  /** Callable each frame; returns current targets. Kept in a ref so RAF reads fresh values. */
  getTargets: () => HoverTarget[]
}

export type HoverState = {
  /** Normalized cursor position (viewport coords, px). null = no visible wrist. */
  cursor: { x: number; y: number } | null
  /** Active dwell target key + progress [0,1]. */
  activeKey: string | null
  progress: number
}

export function useHoverTrigger({
  landmarksRef,
  mirror = true,
  dwellMs = 1200,
  onSelect,
  enabled = true,
  cameraBoundsRef,
  getTargets,
}: UseHoverTriggerOptions): HoverState {
  const [hover, setHover] = useState<HoverState>({ cursor: null, activeKey: null, progress: 0 })
  const onSelectRef = useRef(onSelect)
  const getTargetsRef = useRef(getTargets)
  onSelectRef.current = onSelect
  getTargetsRef.current = getTargets

  useEffect(() => {
    if (!enabled) {
      setHover({ cursor: null, activeKey: null, progress: 0 })
      return
    }
    let raf = 0
    let cancelled = false
    let activeKey: string | null = null
    let activeStart = 0

    const loop = () => {
      if (cancelled) return
      const lms = landmarksRef.current
      const bounds = cameraBoundsRef.current
      let cursor: { x: number; y: number } | null = null

      if (lms && lms.length > 0 && bounds) {
        // 가장 visibility 높은 손목 선택
        const lw = lms[LEFT_WRIST]
        const rw = lms[RIGHT_WRIST]
        const useLw = lw && lw[3] >= VIS_THR && (!rw || lw[3] >= rw[3])
        const useRw = rw && rw[3] >= VIS_THR && (!useLw)
        const wrist = useLw ? lw : useRw ? rw : null
        if (wrist) {
          // normalized → viewport coords (camera bound 기준)
          const nx = mirror ? 1 - wrist[0] : wrist[0]
          const ny = wrist[1]
          cursor = {
            x: bounds.left + nx * bounds.width,
            y: bounds.top + ny * bounds.height,
          }
        }
      }

      let hitKey: string | null = null
      if (cursor) {
        const targets = getTargetsRef.current()
        for (const t of targets) {
          if (
            cursor.x >= t.rect.left &&
            cursor.x <= t.rect.right &&
            cursor.y >= t.rect.top &&
            cursor.y <= t.rect.bottom
          ) {
            hitKey = t.key
            break
          }
        }
      }

      let progress = 0
      const now = performance.now()
      if (hitKey !== null) {
        if (activeKey !== hitKey) {
          activeKey = hitKey
          activeStart = now
        }
        progress = Math.min(1, (now - activeStart) / dwellMs)
        if (progress >= 1) {
          const sel = activeKey
          activeKey = null
          activeStart = 0
          progress = 0
          onSelectRef.current(sel!)
        }
      } else {
        activeKey = null
        activeStart = 0
      }

      setHover({ cursor, activeKey, progress })
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
    }
  }, [landmarksRef, cameraBoundsRef, mirror, dwellMs, enabled])

  return hover
}
