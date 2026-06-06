# CLAUDE.md — `ui_ux/backend/` 작업 컨텍스트

## 폴더 목적
FastAPI 서버. 폰 PWA가 보내는 관절 좌표를 받아 AI 팀 분석 엔진으로 넘기고, 결과를 다시 폰으로 보낸다. 단일 출처는 [`../../docs/PLAN.md`](../../docs/PLAN.md) §5.

## 환경 (검증된 페어)
- **Python 3.13.9** (pyenv shim → uv가 miniconda 3.13 interpreter 자동 선택)
- **uv 0.11.14** + venv `.venv/` (gitignore)
- **fastapi 0.136** / **uvicorn 0.47** (`[standard]` extra) / **pydantic 2.13** / **websockets 16**
- 의존성은 `requirements.txt`로 잠금 (PLAN.md §5가 requirements.txt 명시)

## 함정 6가지
1. **uvicorn cwd 의존성** — `app.main:app` 임포트가 해소되려면 cwd가 `ui_ux/backend/`여야 함. 루트나 다른 디렉토리에서 띄우면 `ModuleNotFoundError: app`.
2. **`uvicorn[standard]` 필수** — `[standard]` extra가 `websockets`를 끌어옴. vanilla `uvicorn`만 깔면 WS 핸들러가 404로 떨어짐.
3. **CORS 미들웨어** — 프론트 dev 서버는 `:5173`, 백엔드는 `:8000`이라 다른 origin. `app/main.py`에서 `CORSMiddleware`로 `allow_origins=["*"]` 적용 중 (사설 IP 가정이라 위험 낮음).
4. **HTTPS는 mkcert 인증서 의존** — 폰 브라우저 `getUserMedia`가 secure context 전용이라 HTTPS 필수. 현재 `certs/longstone-cert.pem`+`longstone-key.pem`을 uvicorn `--ssl-*` 플래그로 로드해 8000 포트 TLS 기동. 인증서 누락·만료 시 uvicorn 부팅 단계에서 실패하니 함정으로 인지(`mkcert <IP>` 재발급 + 폰 루트 CA 신뢰 갱신).
5. **`SessionStore`는 프로세스 메모리** — `--reload`로 코드 수정 시 모든 세션 손실. 통합 검증 시 인지. 실제 영속화는 MVP 범위 아님.
6. **mock `generate_coaching`이 `time.sleep(2)`** — `app/ai_bridge.py:_COACHING_LATENCY_SEC`. polling 동작 시각 확인용 시뮬레이션. 실제 AI 통합 시 반드시 sleep 제거.

## 핵심 명령 (backend/ 안에서)
- 초기 셋업:
  ```sh
  uv venv .venv
  uv pip install -r requirements.txt
  ```
- 개발 기동 (HTTPS):
  ```sh
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload \
    --ssl-keyfile certs/longstone-key.pem --ssl-certfile certs/longstone-cert.pem
  ```
- OpenAPI 자동 문서: `https://localhost:8000/docs` (폰에서는 `https://<노트북IP>:8000/docs`)
- 의존성 추가: `uv pip install <pkg> && uv pip freeze > requirements.txt`

## 엔드포인트 (PLAN.md §5 contract)
| Method | Path | mock 동작 |
|---|---|---|
| GET | `/api/exercises` | 고정 3개 (The Seal / Spine Stretch / Bridging) |
| POST | `/api/sessions` | uuid4 `session_id` 발급 + SessionStore 저장 |
| POST | `/api/sessions/{id}/end` | 종료 마킹 + **`BackgroundTasks`로 `generate_coaching` 위임 후 즉시 `SessionEndResp` 반환** |
| GET | `/api/sessions/{id}/report` | **pending(202 `{status:"pending"}`) / ready(200 `{status:"ready", report:...}`) 분기. unknown은 404** |
| WS | `/ws/pose/{session_id}` | 프레임 수신 → `analyze_frame` mock → 즉시 응답 |

## AI 팀 contract (`app/ai_bridge.py`)
```python
def analyze_frame(landmarks: list[list[float]], state: dict) -> dict
def generate_coaching(session_summary: dict) -> dict
```
현재 둘 다 mock(고정 score 85, 정해진 멘트). `generate_coaching`은 의도적으로 `time.sleep(2)`을 가져 polling UI 시뮬에 사용. W6+ 통합 시 `backend/ai/analyzer.py`에서 같은 시그니처로 실제 구현 → `ai_bridge`의 import 만 교체. **`generate_coaching`은 동기 함수로 둬도 OK** — `app/main.py:end_session`이 `BackgroundTasks`로 thread pool에서 실행하므로 실제 LLM 지연(2-5초)도 클라이언트 응답을 막지 않음.

## 파일 구조
```
app/
├── __init__.py
├── main.py          # FastAPI 앱, CORS, REST 라우트, WS 마운트
├── ws.py            # WebSocket /ws/pose 핸들러
├── sessions.py      # SessionStore (in-memory dict 래퍼)
├── schemas.py       # Pydantic 모델 (PLAN.md §5 스키마 1:1)
└── ai_bridge.py     # analyze_frame / generate_coaching mock
ai/                  # W6+ AI 팀 모듈 통합 위치 (현재 빈 폴더)
certs/               # mkcert 인증서 자리 (현재 빈 폴더, *.pem gitignore)
requirements.txt     # uv pip freeze 산출
```

## 후속 작업
- **AI 팀 모듈 통합** (W7-8): AI 팀이 `ai/analyzer.py` 작성 → `ai_bridge.py`가 `from ai.analyzer import ...`로 교체. mock의 `time.sleep` 제거.
- **정적 파일 서빙** (W11-12): Vite `dist/`를 FastAPI `StaticFiles`로 마운트 → 발표용 단일 서버 배포 (브리핑 항목 9)
- **세션 영속화는 NO** — MVP 범위 아님
