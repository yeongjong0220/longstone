# CLAUDE.md — UI/UX 파트 작업 컨텍스트

## 폴더 목적
ETRI 협업 캡스톤 "온디바이스 AI 자세 교정 코칭 시스템"의 UI/UX 파트 작업 폴더. PLAN.md 옵션 B(폰 PWA + 노트북 AI 서버)를 구현한다. UI/UX 2인(오채민 + 1인) 작업 격리 공간으로, AI 팀 폴더(`min/`)와 분리되어 W6 통합 시점까지 서로 독립.

## 단일 출처 (SOT)
모든 설계·일정·인터페이스 결정은 **[`../docs/PLAN.md`](../docs/PLAN.md)** 가 단일 출처. 변경 시 PLAN.md를 먼저 갱신하고 코드 따라가기.

## 하위 구조
- **[`frontend/`](frontend/CLAUDE.md)** — 폰 브라우저 PWA (Vite + React + TS + Tailwind v4 + shadcn). 자세한 환경/함정은 frontend CLAUDE.md.
- **[`backend/`](backend/CLAUDE.md)** — FastAPI 서버 (uv venv + uvicorn + websockets). 자세한 환경/함정은 backend CLAUDE.md.

## AI 팀과의 contract
AI 팀이 W6+에 `backend/ai/analyzer.py`에 두 함수만 구현하면 통합 완료:
```python
def analyze_frame(landmarks: list[list[float]], state: dict) -> dict: ...
def generate_coaching(session_summary: dict) -> dict: ...
```
현재는 `backend/app/ai_bridge.py`의 mock 구현이 동일 시그니처로 자리잡고 있다. 통합 시 import 만 교체.

WebSocket·REST 메시지 스키마는 PLAN.md §5가 정본. `backend/app/schemas.py`가 1:1 미러.

## 현재 진행 상태 (2026-05-18 기준)
- ✅ 프론트 W1-2 leg: Vite/React/Tailwind v4/shadcn 스캐폴딩 + react-router 5개 화면 placeholder
- ✅ 백엔드 W1-2 leg: FastAPI mock 엔드포인트 4개 REST + 1 WS, in-memory SessionStore
- ✅ **mkcert HTTPS** — 양쪽 dev 서버 TLS 기동. `backend/certs/longstone-{key,cert}.pem`을 backend uvicorn `--ssl-*` 플래그 + Vite `server.https`가 공유. **W2 🎯 마일스톤 도달.**
- ✅ **W3-4 통합 echo 회로** (`f485d0f`): `@mediapipe/tasks-vision` + zustand. `src/lib/api.ts` fetch 래퍼, `useCamera`/`usePoseDetector`/`useWebSocket` 훅, `PoseOverlay` 캔버스, Zustand `sessionStore`. 백엔드 `ws.py`는 `LandmarkFrame`/`CoachingFrame` 스키마 검증 wrap. MediaPipe 모델(`public/models/`)은 git 커밋, WASM(~32MB, `public/mediapipe-wasm/`)은 gitignore + `postinstall`이 `node_modules` → `public/`로 복사. Python WS 클라이언트 round-trip 검증.
- ✅ **W5-6 (`92e77a2`)**: Coaching/Analyzing/Report 카드 UI(shadcn card·progress), 3-2-1 카운트다운, rep 도달 시 자동 종료, [한 세트 더] 흐름. **🎯 W6 마일스톤 도달.**
- ✅ **PWA 변환 (`44f632d`)**: `vite-plugin-pwa`로 manifest/service worker/아이콘 자동화. `navigateFallback=index.html`, `mediapipe-wasm`·`models`은 runtimeCache(precache 제외). `preview` 서버도 mkcert HTTPS. `api.ts`에 5초 fetch timeout + 한글 오프라인 메시지.
- ✅ **LLM 호출 비동기화 (`c36b289`)**: `POST /end`는 `BackgroundTasks`로 `generate_coaching` 위임 후 즉시 `SessionEndResp` 반환. `GET /report`는 pending(202)/ready(200) 분기, unknown은 404. `Analyzing`은 1초 polling + 60초 timeout. mock `generate_coaching`에 `time.sleep(2)` (실제 LLM 통합 시 제거).

## 다음 단계
브리핑 항목 잔여: 4(WS 재시도) / 9(FastAPI StaticFiles로 dist 서빙) / 10(컴포넌트별 랩타임). AI 팀 통합은 `backend/app/ai_bridge.py`의 import 한 줄 교체로 완료되는 상태.

## 공통 함정
- **폰 카메라는 HTTPS 필수** — `getUserMedia`는 secure context에서만 동작. 현재 백엔드는 HTTP만이라 폰에서 실제 카메라 권한 다이얼로그를 받으려면 mkcert 셋업 필요 (`brew install mkcert nss` 시스템 변경 동반).
- **사설 IP 환경 전제** — 인터넷 없는 환경에서 동작이 멘토링 5/14 핵심 제약. 같은 Wi-Fi 또는 노트북 핫스팟 모드(macOS Internet Sharing, 기본 IP `192.168.2.1`)로 폰 접속. PLAN.md §6 참조.
- **AI 팀 폴더 (`../min/`) 건드리지 말 것** — AI 팀과 작업 격리. AI 팀 코드 통합은 W6+에 `backend/ai/`로 import 하는 형태로만.

## 문서·커밋 언어
한국어 기본 (루트 CLAUDE.md 규칙 상속).
