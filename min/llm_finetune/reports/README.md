# reports/

학습 시도·실패·평가 결과를 텍스트로 남기는 폴더. 발표 자료의 객관 증거.

## 자동 생성 파일
- `full_<model_slug>_oom.md` — `train_full.py`가 OOM으로 종료될 때 자동 기록 (예: `full_google__gemma-2-2b-it_oom.md`)

## 수동 작성 파일
- `full_qwen05.md`, `full_hcx05.md` — C 단계 학습 성공 시 학습 곡선·peak VRAM·소요 시간·정성 출력 정리
- `eval.md` — base / Gemma-LoRA / Qwen-full / HCX-full 5개 모델의 추론 결과 비교 표

## 형식 권고
- 마크다운, 한국어
- 상단에 일시·디바이스·VRAM 등 메타데이터 블록
- 학습 곡선(loss)은 표 또는 단순 ascii 그래프로 충분
- 정성 평가는 동일 입력에 대한 출력을 표로 병기
