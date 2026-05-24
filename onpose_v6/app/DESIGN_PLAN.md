# Longstone PWA — AURA 디자인 시스템 이식 진행

> 참조: [`docs/UI_DESIGN_SYSTEM.md`](../docs/UI_DESIGN_SYSTEM.md) (명세), [`docs/UI_ALL_SCREENS.jsx`](../docs/UI_ALL_SCREENS.jsx) (레퍼런스 JSX)
> 본 문서는 *이식 작업 진행 트래킹*용. 디자인 명세 단일 출처는 위 두 문서, UI/UX 파트 전체 단일 출처는 [`docs/PLAN.md`](../docs/PLAN.md).

## 확정된 브랜드 값

| 항목 | 값 |
|---|---|
| Primary | `#4F46E5` (AURA 원본 인디고) |
| 로고 폰트 | `Lobster` (Google Fonts) |
| 본문 폰트 | `Noto Sans KR` (Google Fonts) |
| 앱 이름 | `Longstone` |
| 슬로건 | `POSTURE COACH` |
| 모드 | 라이트 단일 (`.dark` 블록 삭제) |

## 진행 원칙

- **각 Step 끝에서 멈춰** 사용자가 브라우저로 검증 → "다음" 신호 후 다음 Step.
- AURA 제품 화면(옷장/매칭/마이/프리미엄)은 **옮기지 않음**. 디자인 토큰 + 공통 컴포넌트만.
- 기존 5개 화면(Home/Setup/Coaching/Analyzing/Report) **재단은 본 작업 범위 밖**. Step 4의 Home 최소 교체는 시각 검증용.

---

## Step 1. Foundation — 토큰/폰트/배경  ✅ (사용자 검증 대기)

**파일**
- [x] `frontend/src/index.css` — AURA 토큰 추가, shadcn 변수(`--primary`/`--background`/`--card`/`--foreground`/`--border`/`--radius`) 재매핑, `.dark` 블록 삭제, `@theme inline`에 `--color-aura-*` 노출
- [x] `frontend/index.html` — Google Fonts preconnect + `Lobster` & `Noto Sans KR` link, `<meta name="theme-color">` `#4F46E5`, `apple-mobile-web-app-status-bar-style` `default`
- [x] `frontend/src/App.tsx` — 루트 div에 `bg-aura-soft` 그라디언트 + `min-h-screen` + `font-sans`
- [x] `tsc -b --noEmit` 클린 / `pnpm build` 통과 (32.99 kB CSS, 400 kB JS)

**검증 게이트**
- (a) `pnpm dev` 기동 후 모든 라우트(`/`, `/setup`, `/coaching`, `/analyzing`, `/report`)에 라벤더→스카이블루 세로 그라디언트 배경
- (b) DevTools Computed style에서 `--primary: #4F46E5`, `--font-logo: Lobster`
- (c) 콘솔에 폰트/모듈 에러 없음
- (d) 기존 5개 화면 기능 회귀 없음 (shadcn Button/Card/Progress가 자동으로 인디고 톤이 됨 — 의도)

---

## Step 2. PWA Manifest  ✅ (사용자 검증 대기)

**파일**
- [x] `frontend/vite.config.ts` — VitePWA manifest:
  - `name: "Longstone"` (← "Longstone Coach")
  - `theme_color: "#4F46E5"` (← `#0a0a0a`)
  - `background_color: "#E8E4F8"` (← `#0a0a0a`)
  - `short_name`, `description`, `icons[]`는 유지
- [x] `dist/manifest.webmanifest` 빌드 산출물 확인 — 모든 필드 정상 출력

**검증 게이트**
- DevTools → Application → Manifest 패널에 `name: Longstone`, `theme_color: #4F46E5`, `background_color: #E8E4F8`
- Lighthouse PWA score 회귀 없음 (95+)
- PWA 아이콘 자체(Longstone 로고화)는 별도 후속

---

## Step 3. Primitives — `frontend/src/components/aura/`  ✅

**신규 디렉토리.** 기존 `src/components/ui/` (shadcn)와 분리.

- [x] `Glass.tsx` — `<Glass strong?>` 기본/Strong 글래스 카드 (배경/blur/border/shadow)
- [x] `StatusBar.tsx` — 9:41 + 신호/배터리 (라이트/다크 prop, mockup 용도)
- [x] `BackHeader.tsx` — 뒤로가기 + 중앙 타이틀 + 우측 액션 슬롯
- [x] `PageHeader.tsx` — Lobster 32px 타이틀 + 서브타이틀 + 우측 액션 슬롯
- [x] `FAB.tsx` — 우하단 floating action button (Primary 배경)
- [x] `BottomSheet.tsx` — 반투명 오버레이 + 화이트 카드 (24px 24px 0 0)
- [x] `GlassListItem.tsx` — 글래스 카드 안의 리스트 행 (썸네일/제목/설명/값)
- [x] `icons.tsx` — Home/Grid/Scan/User/Camera/Calendar/Plus/Back/ChevronRight/Close/Share/Lock/External (+ HomeFilled)
- [x] `index.ts` — 배럴 export
- [x] `pnpm build` 통과 (CSS 32.99 → 38.44 kB, JS 변동 없음 — 아직 import 전이라 tree-shaken)

> **탭바는 미포함** — 현재 5화면 선형 플로우. 탭 네비게이션 도입 시 추가.

**검증 게이트**
- `pnpm build` (`tsc -b && vite build`) 타입/번들 통과
- 8개 컴포넌트 export 정상 — 시각 확인은 Step 4에서 통합

---

## Step 4. Home 최소 교체 — 시각 검증 데모  ✅ (사용자 검증 대기)

**파일**
- [x] `frontend/src/screens/Home.tsx`:
  - 페이지 타이틀을 `<PageHeader title="Longstone" subtitle="운동을 선택해서 시작하세요." />` (Lobster 32px)
  - 운동 리스트를 단일 `<Glass strong>` 컨테이너 + `<GlassListItem>` 행으로 (AURA "최근 매칭" 패턴)
  - pending 시 다른 카드 opacity-50, 클릭 onClick=undefined
  - 로딩/에러 텍스트는 `text-aura-ter` / 기존 `text-destructive` 유지
- [x] `pnpm build` 통과 (CSS 38.36 kB, JS 401.75 kB — 새 컴포넌트 번들 포함 +1.3 kB)

**검증 게이트**
- Home 화면에서 Lobster `Longstone` 타이틀 + Glass Strong 운동 카드 시각 확인
- 운동 카드 탭 → `/setup` 네비게이트 정상
- Setup/Coaching/Analyzing/Report 기능 회귀 없음 (카메라/WS/Polling)

---

## Step 5. 4화면 AURA 비주얼 재단 — Setup/Coaching/Analyzing/Report  ✅ (사용자 검증 완료)

**파일**
- [x] `frontend/src/screens/Analyzing.tsx` — `<Glass strong>` 카드 안에 스피너+텍스트 묶음, 스피너 `border-aura-mute border-t-aura-primary`, 텍스트 `text-aura-ink`/`text-aura-ter`
- [x] `frontend/src/screens/Setup.tsx` — 헤더 → `<PageHeader title="Setup" subtitle=...>` (운동명 prefix 폴백), 카메라 컨테이너 테두리 `border-aura-glass-border`로 정렬, `bg-background` 제거
- [x] `frontend/src/screens/Report.tsx` — `<PageHeader title="결과 리포트" subtitle={exerciseName}>`, 점수/잘한점/개선점 3섹션 `<Glass strong>` + 코칭 메시지 `<Glass>` (약), "한 세트 더" 버튼 `var(--gradient-cta)` 인디고→블루 그라디언트
- [x] `frontend/src/screens/Coaching.tsx` — 절충안. 헤더 → `<PageHeader>` + 세트 도트 `bg-aura-primary`/`bg-aura-mute`, **카메라/카운트다운 검정 카드 유지** (PoseOverlay 가독성), 정확도/반복/피드백 3카드 shadcn `Card` → `<Glass>` 치환
- [x] `pnpm build` 통과 (CSS 38.28 → 37.80 kB, JS 401.94 → 400.98 kB — shadcn Card 미사용으로 소폭 감소)

**원칙**
- 로직 불변: `useCamera`/`usePoseDetector`/`useWebSocket`/`getReport` 폴링/`createSession`/세션 store 모두 0 변경
- Coaching 절충안: 운동 중 카메라 영상 + PoseOverlay 대비 가독성을 디자인 일관성보다 우선 (검정 카드 유지)
- `bg-background` 4화면에서 모두 제거 → App 루트 `bg-aura-soft` 그라디언트가 그대로 보임

**검증 게이트** (모두 완료)
- (a) Analyzing: Glass 카드 + 인디고 스피너 + 2초 후 `/report` 자동 전환
- (b) Setup: Lobster `Setup` 타이틀, 카메라 + 33관절 카운터 정상, "운동 시작" 토글 정상
- (c) Report: Lobster `결과 리포트`, 4 Glass 카드, CTA 그라디언트 버튼, "한 세트 더" → `/setup` 이동
- (d) Coaching: Lobster `${exerciseName}`, 인디고 세트 도트, 3-2-1 카운트다운, WS 실시간 갱신, rep 도달 자동 종료

---

## 후속 작업 (본 plan 범위 밖)

- [x] Setup/Coaching/Analyzing/Report 4화면 AURA 비주얼 재단 (Step 5)
- [x] PWA 아이콘 교체 — `favicon.svg`를 Longstone L 모노그램(인디고 그라디언트 + 흰색 cursive L)으로 교체 후 `@vite-pwa/assets-generator`로 5개 PNG + favicon.ico 일괄 재생성. maskable/apple 패딩 배경은 `#4F46E5`로 명시.
- [x] iOS 스플래시 (`apple-touch-startup-image`) 디자인 후 추가 — Lobster "Longstone" 워드마크 단색 인디고 배경. 8개 디바이스 × portrait+landscape = 14 PNG. `pwa-assets-splash.config.ts` 별도 config (preset에 transparent/maskable/apple sizes 빈 배열로 splash만). index.html에 16개 link 태그(generator 출력의 `-light-` 버그 sed 처리). vite.config includeAssets glob `apple-splash-*.png`.
- [ ] 온보딩 페이지 신설 여부 결정
- [x] (선택) apple-touch-icon 패딩 축소 — `pwa-assets.config.ts`의 apple preset에 `padding: 0.05` 명시. inner 라운드 사각 거의 사라지고 L이 가득.
- [x] (선택) Lobster "L" 정확 복제 — `frontend/scripts/fonts/Lobster-Regular.ttf` (OFL) + `opentype.js`로 글리프 path 추출. `scripts/extract-lobster-l-path.mjs` 일회성 스크립트가 favicon.svg 자동 생성. opentype.js v2의 `toPathData({decimalPlaces:2})` + `<path transform="matrix(1 0 0 -1 0 512)">`로 y-flip 보정.

---

## 진행 로그

| Step | 상태 | 완료일 | 비고 |
|---|---|---|---|
| 1. Foundation | ✅ 검증 대기 | 2026-05-24 | 빌드 클린 통과, 브라우저 시각 검증 필요 |
| 2. Manifest | ✅ 검증 대기 | 2026-05-24 | 빌드 manifest.webmanifest 정상, DevTools Application 확인 필요 |
| 3. Primitives | ✅ 완료 | 2026-05-24 | 8 파일 + 배럴, 빌드 통과. 시각 확인은 Step 4에서 |
| 4. Home 적용 | ✅ 검증 대기 | 2026-05-24 | 빌드 통과, 브라우저 시각/기능 검증 필요 |
| 5. 4화면 재단 | ✅ 검증 완료 | 2026-05-24 | Analyzing/Setup/Report/Coaching(절충안). 빌드 통과, 4화면 모두 브라우저 검증 OK |
| 후속1. PWA 아이콘 | ✅ 검증 대기 | 2026-05-24 | `favicon.svg` Longstone L 모노그램 신규 작성, `pnpm generate-pwa-assets` 5 PNG + favicon.ico 재생성, build 통과. DevTools Application/홈 화면 추가 시각 검증 필요 |
| 후속1-polish. 아이콘 정밀화 | ✅ 검증 대기 | 2026-05-24 | apple padding 0.05 + Lobster.ttf opentype.js 추출로 글리프 fidelity 확보. `extract-lobster-l-path.mjs` 일회성 스크립트 commit. build 통과 |
| 후속2. iOS 스플래시 | ✅ 검증 대기 | 2026-05-24 | Lobster "Longstone" 워드마크 splash, 8 디바이스 × portrait+landscape = 14 PNG. `pwa-assets-splash.config.ts` + `build-splash-source.mjs`. index.html 16 link 추가. precache 528→669 KiB |
| 후속2-polish. PWA UX 통일 | ✅ 검증 대기 | 2026-05-24 | (a) opentype.js NaN segment 정규식 제거로 splash 'e' wisp 해결, favicon에도 동일 적용. (b) status-bar-style `default` → `black-translucent` + App.tsx 루트에 `paddingTop: env(safe-area-inset-top)` 추가 → 본문 라벤더 그라디언트가 status bar 영역까지 자연스럽게 확장, 콘텐츠는 safe-area 아래에서 시작 |
