# CLAUDE.md — `ui_ux/frontend/` 작업 컨텍스트

## 폴더 목적
폰 브라우저에서 도는 PWA. 카메라 프레임은 폰을 떠나지 않고, MediaPipe.js로 추출한 관절 좌표만 WebSocket으로 백엔드에 송신한다. 단일 출처는 [`../../docs/PLAN.md`](../../docs/PLAN.md) §4.

## 환경 (검증된 페어, 변경 시 회귀 위험)
- **pnpm 11.1.2** (`~/Library/pnpm`, PATH는 `~/.zshrc:146-152`에 자동 추가)
- **Vite 8.0** / **React 19.2** / **TypeScript 6.0**
- **Tailwind v4.3** + `@tailwindcss/vite` 플러그인 (PostCSS·tailwind.config.js 둘 다 **없음**)
- **shadcn 4.7** — preset `nova`, base `radix`, 아이콘 `lucide-react`, 폰트 `@fontsource-variable/geist`
- **react-router-dom 6.30** — v7 X (PLAN.md §4가 v6 명시)
- **vite-plugin-pwa 1.3.0** + **@vite-pwa/assets-generator 1.0.2** — manifest/SW/아이콘 자동화. sharp native 빌드 의존이라 `pnpm-workspace.yaml`의 `onlyBuiltDependencies`에 `sharp` 포함.
- 디자인 토큰은 `src/index.css`에 OKLCH 색공간으로 주입됨 (shadcn init 산출)

## 함정 8가지
1. **TypeScript 7부터 `baseUrl` deprecated** — `tsconfig.app.json`/`tsconfig.json`에 **`paths`만** 두고 `baseUrl`은 제거 상태. 추가하면 빌드 깨짐(`TS5101`).
2. **shadcn init은 `pnpm-workspace.yaml`에 placeholder 남김** — 처음 한 번 `msw: set this to true or false` 라인이 생기는데 그대로 두면 다음 install이 yaml 파싱 에러. `onlyBuiltDependencies: [esbuild, sharp]`로 교체 후 init을 `--force --no-reinstall`로 재시도.
3. **`pnpm install -g`는 `/usr/local/lib` 권한 막힘** — sudo 없이 가려면 standalone 스크립트:
   ```sh
   curl -fsSL https://get.pnpm.io/install.sh | sh -
   ```
4. **shadcn 컴포넌트는 항상 `@/components/ui/*` import** — alias가 `tsconfig.app.json:paths`와 `vite.config.ts:resolve.alias` 양쪽에 동시 설정돼야 둘 다(TS·런타임) 해결됨.
5. **`vite.config.ts`의 `server.host: true`** — 끄면 `pnpm dev`가 Network URL을 출력하지 않아 폰에서 못 붙음. 끄지 말 것. `preview`도 동일 cert + `host: true` 설정.
6. **HTTPS는 인증서 파일 존재가 조건** — `vite.config.ts`가 `../backend/certs/longstone-{key,cert}.pem` 둘 다 있을 때만 `server.https`/`preview.https`를 켠다. 누락 시 자동으로 HTTP로 떨어져 폰 카메라 차단됨.
7. **mkcert cert SAN이 IP 매칭** — 노트북 IP가 바뀌면(장소 이동 등) cert 재발급 필요. `mkcert -cert-file longstone-cert.pem -key-file longstone-key.pem localhost 127.0.0.1 <새IP> 192.168.2.1` + 서버 재기동. 폰의 root CA 신뢰는 그대로 유지됨(leaf 변경뿐).
8. **dev mode SW ≠ production SW** — `vite-plugin-pwa`의 `devOptions.enabled: true`는 SW 등록만 시연. Workbox precache는 **production 빌드에서만** 동작. 오프라인 검증은 반드시 `pnpm build && pnpm preview` (preview도 mkcert HTTPS).

## 핵심 명령 (frontend/ 안에서)
- 개발 서버: `pnpm dev` → Local `https://localhost:5173/` + Network URL (`vite.config.ts`가 `../backend/certs/longstone-*.pem`을 자동 로드 → HTTPS 기동)
- 폰 접속 IP 확인: `ipconfig getifaddr en0` → 폰 Safari `https://<해당IP>:5173`
- 프로덕션 빌드: `pnpm build` → `dist/`
- shadcn 컴포넌트 추가: `pnpm dlx shadcn@latest add <component>` (예: `add card input dialog`)
- 의존성 추가: `pnpm add <pkg>` / dev면 `-D`

## 현재 라우트 (`src/App.tsx`)
- `/` **Home** — `listExercises` 호출, 카드 선택 → `createSession` → store에 저장 → `/setup`
- `/setup` **Setup** — `useCamera` + `usePoseDetector`, 33관절 인식 시 [운동 시작] 활성화
- `/coaching` **Coaching** — 3-2-1 카운트다운 → `useWebSocket` 좌표 송신 + `CoachingFrame` 수신, 정확도/반복/피드백 카드 갱신, rep 도달 또는 [운동 종료] 시 `endSession` → `/analyzing`
- `/analyzing` **Analyzing** — `getReport` 1초 polling, ready 시 `/report` (60초 timeout 시 에러 + [홈으로])
- `/report` **Report** — `Report` 표시 + [한 세트 더](새 세션 + `/setup`) / [홈으로]

## 디렉토리 골격 (PLAN.md §4)
```
src/
├── screens/        # 5개 화면 컴포넌트 (라우트별 1파일)
├── components/     # 공통 UI (스켈레톤 캔버스 등 — W3+)
├── components/ui/  # shadcn 복붙 컴포넌트 (CLI가 채움)
├── stores/         # Zustand 스토어 (W3+)
├── hooks/          # useCamera, usePoseDetector, useWebSocket (W3-6)
├── lib/            # utils.ts (shadcn cn 헬퍼), MediaPipe 초기화 (W3-4)
└── App.tsx         # 라우팅
```

## 후속 작업 (남은 것만)
- ✅ W3-4 MediaPipe + WebSocket, W5-6 코칭 메인 UI, W11 PWA(manifest/SW/아이콘), 카메라 권한 거부 UX, 홈 횟수·세트 stepper, 전문가 영상 풀스크린 오버레이 완료
- WS 끊김 5회 재시도 + 에러 토스트 (`useWebSocket`) — PLAN.md §5
- Framer Motion 화면 전환 (W9-10 polish)
- 컴포넌트별 랩타임 측정 (멘토 영종 요청, W11)
