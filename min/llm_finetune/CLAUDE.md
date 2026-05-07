# CLAUDE.md — LLM 파인튜닝 작업 컨텍스트

## 폴더 목적
필라테스 코칭 LLM 파인튜닝 인프라. 교수 피드백("full FT 우선, 정 안 되면 LoRA")을 실증한 결과: Gemma-2-2b는 환경상 full FT 불가(증거 `reports/full_google__gemma-2-2b-it_oom.md`), 0.5B 모델은 full FT 가능. 사용 가이드는 [README.md](README.md), 이 문서는 작업 컨텍스트(주의사항·함정).

## 환경 (검증된 페어, 변경 시 segfault 위험)
- Python 3.10 / venv: `.venv/` (gitignore)
- torch 2.11.0+cu126 — PyPI 기본 wheel은 CPU only이므로 `--index-url https://download.pytorch.org/whl/cu126` 명시 필수
- transformers 4.46.0 / trl 0.12.2 / peft 0.13.2 / accelerate 1.0.1
- bitsandbytes **0.45.0+** (cu126 binary 포함, 0.44 이하는 8bit Adam 호출 시 NameError)
- datasets 3.6.0 + pyarrow 19.0.1 (datasets 4.x + pyarrow 24.x 조합은 native ABI segfault)

## 작업 시 함정 3가지
1. **`PYTHONUTF8=1` 또는 `-X utf8` 필수** — Windows cp949 환경에서 trl이 jinja 템플릿(deepseekv3.jinja 등)을 인코딩 미명시로 read하다가 UnicodeDecodeError. 모든 학습/추론 명령에 prefix.
2. **Gemma 라이선스 동의** — `huggingface.co/google/gemma-2-2b-it` 페이지에서 사용자가 직접 동의해야 함. 토큰만으로는 403. metadata API는 통과해도 weight download는 따로 막힘.
3. **HF cache 디스크 압박** — Gemma 5GB + 0.5B 모델 1GB×2 + 양자화 cache. 데스크탑 잔여 ~25GB라 본 학습 시 부족 가능. `HF_HOME` 외장으로 옮길지 미리 결정.

## 핵심 명령 (PYTHONUTF8=1 prefix 가정)
- 환경 점검: `python env_check.py`
- Full FT (LoRA 없이): `python train_full.py --base-model <id> --data data/<file> --output checkpoints/<dir>` — OOM 시 `reports/full_<slug>_oom.md` 자동 생성
- QLoRA (어댑터): `python train_lora.py --data data/<file> --output checkpoints/<dir>`
- 추론: `python infer.py --base-model <id-or-path> [--adapter <path>] --input-file data/<file> --n 3`

## 단계별 검증 결과 (2026-05-03, dummy 6쌍 1 epoch 기준)
| 단계 | 결과 | 산출물 |
| --- | --- | --- |
| A. Gemma full FT | ❌ OOM peak **14.5GB** vs VRAM 8GB | `reports/full_google__gemma-2-2b-it_oom.md` |
| C. Qwen0.5 full FT | ✅ loss 3.29 | `checkpoints/full_qwen05/` |
| C. HCX0.5 full FT | ✅ loss 2.89 (한국어 가장 매끄러움) | `checkpoints/full_hcx05/` |
| LoRA. Gemma QLoRA | ✅ loss 3.31, 어댑터 25MB (0.24% 학습) | `checkpoints/lora_gemma/` |
| Eval | 5종 비교, 시드 톤 학습은 LoRA가 가장 명확 | `reports/eval.md` |

## 데이터 스키마 truth source
- v3 입력 분포 (구버전, dummy.jsonl 시드만 이 형태): `min/min_dev_park/live_ai_coach_v3_api.py:271-275`
- v4 입력 분포 (sft.jsonl 본 학습): `live_ai_coach_v4_bundle/min/min_dev_park/live_ai_coach_v4_quality.py:261-281` + `live_ai_coach_v4_bundle/pilates_temporal_lifter/runtime_quality.py:124-136` 의 `format_summary_for_prompt`
- 학습 데이터 형식: [data/schema.md](data/schema.md) (Alpaca instruction/input/output, v3·v4 두 분포 명세)

## 합성 파이프라인 (`data/synth/`)
본 학습용 `data/sft.jsonl` 은 v4 분포로 합성한다. 4단계 스크립트를 순서대로 실행:

```bash
cd min/llm_finetune
python data/synth/build_inputs.py                          # 약 500개 input → data/synth/inputs.jsonl + synth_seed.json
python data/synth/generate_outputs.py                      # Gemini 2.5 Flash 호출 → data/synth/raw_outputs.jsonl
python data/synth/filter.py                                # 자동 게이트 → data/synth/filtered.jsonl + reports/synth_review_queue.md
python data/synth/split.py                                 # train/eval stratified → data/sft.jsonl + data/eval.jsonl
```

- `generate_outputs.py` 는 `GOOGLE_API_KEY` 환경변수 또는 `live_ai_coach_v4_bundle/.env` 에서 키를 읽는다. resumable 이라 중단되어도 다시 실행하면 누락된 id 만 호출.
- dry-run: `build_inputs.py --limit 2` (자세별 2개) → `generate_outputs.py --limit 6` 으로 6쌍만 생성.
- `synth_seed.json` 은 git 추적 (재현용), `inputs.jsonl`·`raw_outputs.jsonl`·`filtered.jsonl`·`sft.jsonl`·`eval.jsonl` 는 모두 gitignore.

## 산출물 / git 관리
- `checkpoints/` (gitignore) — 학습 모델·어댑터
- `reports/` (tracked) — OOM 로그·평가 결과는 발표 증거라 추적
- `data/sft.jsonl` (gitignore) / `data/schema.md`·`data/dummy.jsonl` (tracked)
- `.venv/` (gitignore)

## 후속 작업
- ✅ 합성 SFT 데이터셋 plan: `~/.claude/plans/llm-sequential-kernighan.md`. 파이프라인은 위 "합성 파이프라인" 섹션 참고.
- 본 학습 (HCX-full + Gemma-LoRA A/B 비교): `python train_full.py --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B --data data/sft.jsonl --output checkpoints/full_hcx05_v4` 와 `python train_lora.py --data data/sft.jsonl --output checkpoints/lora_gemma_v4` 를 같은 데이터셋으로.
- A/B 평가 결과: `reports/eval_v4_dataset.md` 에 기록.
- `min/min_dev_park/live_ai_coach_v4_quality.py` 의 `request_feedback` 을 학습된 로컬 모델 호출로 교체 (Gemini 의존 제거).
