# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트
온디바이스 AI 자세 교정 코칭 시스템 캡스톤 (ETRI 협업, 2026-1학기). 라이브 카메라 영상에서 운동 자세를 실시간 분석하고 피드백을 제공. 클라우드 의존 없는 추론이 목표.

## 현재 작업 위치
- **`min/min_dev_park/`** — 현재 진행 중인 메인 코드. 최신 엔트리포인트는 `live_ai_coach_v3_api.py` (v2 → v3 순으로 발전).
- **`min/preproccessing/`** — 데이터 전처리 파이프라인. 파일명 숫자 prefix(`00_` → `01_` → `02_`)가 실행 순서. 폴더명 오타(preproccessing)는 의도된 것이니 변경하지 말 것.
- **`dev/`, `coals_EDA/`** — 초기 프로토타입. 참고용이며 더 이상 활성 개발 대상 아님.
- **`docs/`** — 발표자료(PDF/PPTX)와 설계 문서. `pipeline.md`가 아키텍처 단일 출처.

## 아키텍처 (docs/pipeline.md 기준)
입력(라이브 카메라 또는 AI Hub 3D keypoint CSV)
→ MediaPipe Pose로 3D keypoint 추출
→ mid-hip 기준 body-centered 좌표계 정규화
→ Phase Detector: 키네마틱 이벤트(angle extrema, 각속도 방향, expert band 진입/이탈)로 준비/진입/핵심자세/복귀 phase 판별 — **고정 시간 분할 아님**
→ Kinematics Engine: 관절각, ROM, 좌우대칭, 각속도 계산
→ phase별 기준 범위와 비교해 즉시 정상/주의/오류 시각 피드백
→ rep/세트 종료 시 오류 통계를 템플릿 또는 소형 LLM(Gemma-2-2b 4-bit)에 전달해 자연어 코칭 생성

데이터셋은 AI Hub Pilates "Spine Stretch" 등. 2D 투영 깊이 모호성 때문에 side-view 카메라(CAM 메타데이터)를 우선 사용.

## 스택 / 실행
Python · MediaPipe Pose · YOLOv8 · OpenCV · transformers(Gemma-2-2b 4-bit). `requirements.txt`/`pyproject.toml`/Makefile/테스트 프레임워크 모두 **없음**. 스크립트는 `python <파일>.py`로 단독 실행하며 대부분 웹캠 입력 기반.

## 작업 시 주의
- 문서·커밋·논의 언어는 한국어가 기본.
- 새 코드는 `min/` 트리에 추가; `dev/`·`coals_EDA/`는 건드리지 말 것(제안 시 사용자에게 확인).
- phase 분할 로직을 수정할 때는 `docs/pipeline.md`의 키네마틱 이벤트 정의를 먼저 확인.
