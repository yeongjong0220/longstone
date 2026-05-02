# LLM 파인튜닝 작업 폴더

자세 교정 코칭(`live_ai_coach_v2.py`/`v3_api.py`)에 쓰이는 자연어 코칭 모듈을 파인튜닝하기 위한 인프라.

## 목적과 흐름

중간점검 교수 피드백을 정확히 반영: **(1) full fine-tuning을 1차로 시도, (2) 환경상 안 되면 LoRA로 fallback**.

RTX 3050 8GB 환경에서:
- Gemma-2-2b 같은 큰 모델 → full FT 불가능(시도 후 OOM 로그 확보) → **QLoRA fallback**
- Qwen2.5-0.5B / HyperCLOVAX-0.5B 같은 작은 모델 → **full FT 직접 수행**

따라서 이 폴더에는 두 학습 스크립트가 공존:
- `train_full.py` — LoRA 없이 full SFT
- `train_lora.py` — 4-bit QLoRA SFT

데이터(`data/sft.jsonl`)는 별도 단계에서 준비. 이번 단계는 인프라/환경/시도 로그 확보가 목표.

## 폴더 구조
```
min/llm_finetune/
├─ README.md            # 이 문서
├─ requirements.txt
├─ env_check.py         # CUDA / VRAM / 4-bit 로딩 점검
├─ data/
│  ├─ schema.md         # SFT 데이터 스키마 (live_ai_coach_v3_api.py 분포와 일치, 다중 자세)
│  ├─ dummy.jsonl       # dry-run 전용 6쌍 (자세별 2건씩, 학습 결과 품질 보장 X)
│  └─ sft.jsonl         # ⚠️ 미생성. 실제 학습용 (gitignore)
├─ train_full.py        # Full SFT (LoRA X) — 큰 모델은 OOM 로그를 reports/에 남김
├─ train_lora.py        # QLoRA SFT (4-bit + LoRA)
├─ infer.py             # base / base+adapter / full FT 모델 추론
├─ reports/             # OOM 로그, 학습 결과, 평가 비교
└─ checkpoints/         # 학습 산출물 (gitignore)
```

## 환경
- 데스크탑: RTX 3050 8GB · i5-12400F · RAM 16GB · 디스크 잔여 ~36GB
- Python 3.10+ 권장
- CUDA 11.8 또는 12.x

## 설치
```powershell
cd min/llm_finetune
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
`bitsandbytes` Windows 빌드 이슈 시 `pip install bitsandbytes==0.43.3` 등 핀 재시도.

## 단계별 실행 (현재 plan)

### 0. 환경 점검
```powershell
python env_check.py
```
처음 실행 시 Gemma-2-2b-it(~5GB)이 Hugging Face 캐시에 다운로드된다. 디스크 부족 시 `HF_HOME` 환경변수로 큰 드라이브로 이동.

### A. 큰 모델 full FT 시도 (OOM 예상)
```powershell
python train_full.py --base-model google/gemma-2-2b-it --data data/dummy.jsonl --output checkpoints/full_gemma_attempt --epochs 1
```
- 예상 결과: `torch.cuda.OutOfMemoryError`
- 자동으로 `reports/full_google__gemma-2-2b-it_oom.md` 생성
- 이 로그가 "환경상 full FT 불가 → LoRA fallback 정당화"의 객관 증거

### C. 작은 모델 full FT (예상 성공)
```powershell
python train_full.py --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/dummy.jsonl --output checkpoints/full_qwen05 --epochs 1
python train_full.py --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B --data data/dummy.jsonl --output checkpoints/full_hcx05 --epochs 1
```
성공 시 모델 가중치 + 토크나이저가 `checkpoints/full_*`에 저장됨.

### LoRA fallback (Gemma-2-2b QLoRA)
```powershell
python train_lora.py --data data/dummy.jsonl --output checkpoints/lora_gemma --epochs 1
```
산출물은 어댑터(`adapter_model.safetensors`) 1개. base는 추론 시 별도로 로드.

### 비교 추론
```powershell
# base
python infer.py --base-model google/gemma-2-2b-it --input-file data/dummy.jsonl --n 3
python infer.py --base-model Qwen/Qwen2.5-0.5B-Instruct --input-file data/dummy.jsonl --n 3

# full FT 결과 (--base-model에 로컬 경로 지정 가능)
python infer.py --base-model checkpoints/full_qwen05 --input-file data/dummy.jsonl --n 3
python infer.py --base-model checkpoints/full_hcx05 --input-file data/dummy.jsonl --n 3

# LoRA
python infer.py --base-model google/gemma-2-2b-it --adapter checkpoints/lora_gemma --input-file data/dummy.jsonl --n 3
```
정성 결과를 `reports/eval.md`에 표로 기록.

## 학습 설정 비교

### `train_full.py` (RTX 3050 8GB 기준)
- bf16 학습 + gradient checkpointing
- `optim="adamw_bnb_8bit"` (bitsandbytes 8-bit Adam)
- batch 1 · grad_accum 8 · lr 2e-5 · max_seq 384
- OOM 발생 시 `reports/full_<slug>_oom.md` 자동 생성 후 exit 2

### `train_lora.py` (QLoRA)
- 4-bit NF4 + double quant + bfloat16 compute
- LoRA `r=16, alpha=32, dropout=0.05`, target = `q/k/v/o_proj`
- batch 1 · grad_accum 16 · lr 2e-4 · max_seq 384

OOM 시 완화 순서: `--max-seq 256` → `--lora-r 8` → `target_modules`에서 `o_proj` 제외.

## 학습 데이터 스키마
자세한 정의는 [`data/schema.md`](data/schema.md). 핵심 필드 3개:
- `instruction`: 모든 샘플 동일한 시스템 지시문(필라테스 강사 페르소나)
- `input`: 동작명(`name_en`/`name_kr`) + error_frames + 자세별 ref_hip/ref_knee + 평균값 + state로 채워진 통계 문자열 (`live_ai_coach_v3_api.py`의 `request_feedback()` 프롬프트와 동일 분포)
- `output`: 한국어 코칭 응답 (2문장 이내, 자세 특성 반영)

지원 자세: `The_Seal`(더 씰), `Spine_Stretch`(스파인 스트레치), `Bridging`(브릿징).

## 후속 작업 (이 폴더 밖)
- 합성 SFT 데이터셋 구축 (시드 작성 → 외부 LLM augment → 사람 검수)
- 본 학습 (sft.jsonl 기반)
- `live_ai_coach_v2.py`(로컬 Gemma)의 `model.generate()`에 `PeftModel.from_pretrained` 통합 → 어댑터 swap

## 참고 파일
- 입력 분포 truth source: `min/min_dev_park/live_ai_coach_v3_api.py:271-275` (`request_feedback`의 llm_prompt)
- 로컬 추론 진입점: `min/min_dev_park/live_ai_coach_v2.py:228-234` (어댑터 swap 대상)
- 자세 정의(POSE_CONFIG): `min/min_dev_park/live_ai_coach_v3_api.py:61-83`
- 키네마틱 메트릭: `min/preproccessing/02_feature_engineering.py`
