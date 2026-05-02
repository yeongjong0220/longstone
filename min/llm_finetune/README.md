# LLM 파인튜닝 작업 폴더

자세 교정 코칭(`live_ai_coach_v2.py`)에 쓰이는 자연어 코칭 모듈을 **Gemma-2-2b-it 기반 QLoRA로 파인튜닝**하기 위한 인프라.

## 목적
중간점검 교수 피드백("파운데이션 모델 그대로 쓰지 말고 LoRA라도 돌려라") 반영.
이번 단계에서는 **데이터만 들어오면 한 줄 명령으로 학습이 돌아가는 환경**까지 끝낸다.
실제 학습 데이터(`data/sft.jsonl`) 확보, base vs LoRA 평가, 실시간 파이프라인 어댑터 통합은 후속 작업.

## 폴더 구조
```
min/llm_finetune/
├─ README.md            # 이 문서
├─ requirements.txt
├─ env_check.py         # CUDA / VRAM / 4-bit 로딩 점검
├─ data/
│  ├─ schema.md         # SFT 데이터 스키마 정의 (live_ai_coach_v2.py 분포와 일치)
│  ├─ dummy.jsonl       # dry-run 전용 5쌍 (학습 결과 품질 보장 X)
│  └─ sft.jsonl         # ⚠️ 미생성. 실제 학습용 (gitignore)
├─ train_lora.py        # QLoRA 학습 본체
├─ infer.py             # base / base+adapter 단일 추론
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

`bitsandbytes`는 Windows 빌드 이슈가 종종 있다. 실패 시 `pip install bitsandbytes==0.43.3` 등 핀 후 재시도.

## 단계별 실행

### 1. 환경 점검
```powershell
python env_check.py
```
처음 실행 시 Gemma-2-2b-it(~5GB)이 HuggingFace 캐시에 다운로드된다. 잔여 디스크 부족 시 `HF_HOME` 환경변수로 외장/큰 드라이브로 옮겨라.

### 2. dry-run (데이터 없이 파이프라인만 검증)
```powershell
python train_lora.py --data data/dummy.jsonl --output checkpoints/dryrun --epochs 1
```
- loss가 떨어지면서 `checkpoints/dryrun/adapter_model.safetensors`가 생기면 통과.
- dummy.jsonl은 5쌍밖에 없으므로 학습 결과가 의미 있는 코칭은 아니다.

### 3. 추론 비교
```powershell
# base만
python infer.py --input-file data/dummy.jsonl --n 3

# adapter 적용
python infer.py --input-file data/dummy.jsonl --adapter checkpoints/dryrun --n 3
```
응답시간(ms) · peak VRAM이 같이 출력된다.

### 4. 실제 학습 (데이터 확보 후)
`data/sft.jsonl`을 `data/schema.md` 스키마대로 채운 뒤:
```powershell
python train_lora.py --data data/sft.jsonl --output checkpoints/spine_seal_v1 --epochs 3
```

## QLoRA 설정 (RTX 3050 8GB 기준)
`train_lora.py`의 기본값:
- 4-bit NF4 + double quant + bfloat16 compute
- LoRA `r=16, alpha=32, dropout=0.05`, target = `q/k/v/o_proj`
- batch 1 · grad_accum 16 · cosine lr=2e-4 · max_seq 384
- gradient checkpointing 활성

OOM이 뜨면 다음 순서로 완화:
1. `--max-seq 256`
2. `--lora-r 8`
3. target_modules에서 `o_proj` 제외 (`train_lora.py` LoraConfig 수정)

## 학습 데이터 스키마
자세한 정의는 [`data/schema.md`](data/schema.md). 핵심 필드 3개:
- `instruction`: 모든 샘플 동일한 시스템 지시문(필라테스 강사 페르소나)
- `input`: 동작명(`name_en`/`name_kr`) + error_frames + 자세별 ref_hip/ref_knee + 평균값 + state로 채워진 통계 문자열 (`live_ai_coach_v3_api.py`의 `request_feedback()` 프롬프트와 동일 분포)
- `output`: 한국어 코칭 응답 (2문장 이내, 자세 특성 반영)

지원 자세: `The_Seal`(더 씰), `Spine_Stretch`(스파인 스트레치), `Bridging`(브릿징). POSE_CONFIG에 자세를 추가하면 input 변수만 채워 학습 샘플 추가 가능.

## 후속 작업 (이 폴더 밖)
- 합성 SFT 데이터셋 구축 (시드 작성 → 외부 LLM augment → 사람 검수)
- base vs LoRA 정량/정성 평가 리포트
- `live_ai_coach_v2.py`(로컬 Gemma 추론)의 `model.generate()`에 `PeftModel.from_pretrained` 통합 → 어댑터 swap. 단 v3_api는 Gemini API 호출이라 어댑터 swap 대상 아님 — 로컬 추론 경로에서만 적용.

## 참고 파일
- 입력 분포 truth source: `min/min_dev_park/live_ai_coach_v3_api.py:271-275` (`request_feedback`의 llm_prompt)
- 로컬 추론 진입점: `min/min_dev_park/live_ai_coach_v2.py:228-234` (어댑터 swap 대상)
- 자세 정의(POSE_CONFIG): `min/min_dev_park/live_ai_coach_v3_api.py:61-83`
- 키네마틱 메트릭 정의: `min/preproccessing/02_feature_engineering.py`
