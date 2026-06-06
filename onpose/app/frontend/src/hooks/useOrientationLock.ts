import { useEffect } from "react"

export type OrientationTarget = "portrait" | "landscape"

type ScreenOrientationLockable = ScreenOrientation & {
  lock?: (type: OrientationLockType) => Promise<void>
  unlock?: () => void
}

/**
 * 화면을 target orientation으로 잠금 시도.
 * - `screen.orientation.lock()`은 Android Chrome standalone PWA 등에서만 성공한다.
 * - iOS Safari/PWA는 이 API를 지원하지 않으므로 실패하며, 그 경우 가로 전용 화면은
 *   `<RotateHint/>`가 "가로로 돌려주세요" 안내를 띄운다 (강제 CSS 회전은 쓰지 않음).
 */
export function useOrientationLock(target: OrientationTarget): void {
  useEffect(() => {
    const orientation = (window.screen?.orientation ?? null) as ScreenOrientationLockable | null
    const lockType: OrientationLockType =
      target === "portrait" ? "portrait-primary" : "landscape-primary"

    if (orientation?.lock) {
      orientation.lock(lockType).catch(() => {
        // 미지원 환경(iOS 등)에서는 실패 — RotateHint가 안내를 담당.
      })
    }

    return () => {
      orientation?.unlock?.()
    }
  }, [target])
}
