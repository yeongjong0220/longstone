import { PhoneFrame } from "@/components/PhoneFrame"
import { FitnessLevel } from "@/screens/FitnessLevel"
import { Home } from "@/screens/Home"
import { Splash } from "@/screens/Splash"
import { Stats } from "@/screens/Stats"

const SCREENS = [
  { label: "1 · Splash", node: <Splash /> },
  { label: "2 · Fitness level", node: <FitnessLevel /> },
  { label: "3 · Home", node: <Home /> },
  { label: "4 · Stats", node: <Stats /> },
]

/** 캡쳐 레퍼런스(GO Club) 4화면을 폰 프레임에 나란히 렌더하는 테스트 갤러리. */
export default function Gallery() {
  return (
    <div className="min-h-full w-full bg-[#050507] px-8 py-12">
      <header className="mx-auto mb-10 max-w-5xl text-center">
        <h1 className="font-sans text-2xl font-bold text-white">
          GO Club — UI 재현 테스트
        </h1>
        <p className="mt-2 text-sm text-white/45">
          본 프로젝트와 분리된 격리 미니앱 · Vite + React 19 + TS + Tailwind v4 ·
          포트 5174
        </p>
      </header>

      <div className="no-scrollbar flex justify-start gap-7 overflow-x-auto px-2 pb-4 lg:justify-center">
        {SCREENS.map((s) => (
          <PhoneFrame key={s.label} label={s.label}>
            {s.node}
          </PhoneFrame>
        ))}
      </div>
    </div>
  )
}
