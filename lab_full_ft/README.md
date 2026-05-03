# lab_full_ft — 연구실 GPU full fine-tuning 핸드오프

캡스톤(온디바이스 자세 교정 코칭) LLM 파인튜닝을 연구실 GPU에서 재현·실행하기 위한 자기완결 패키지입니다.
사용자 데스크탑(RTX 3050 8GB)에서 검증된 코드·환경·더미 데이터를 그대로 동결한 snapshot이며, 산출물(checkpoint·loss·OOM 로그)은 채민에게 회수하는 것을 전제로 합니다.

## 0. 폴더 구성

```
lab_full_ft/
├─ README.md           # 이 문서
├─ environment.yml     # conda env 정의
├─ requirements.txt    # pip 의존성 (검증된 페어)
├─ env_check.py        # 환경 점검 스크립트
├─ train_full.py       # full fine-tuning 학습 스크립트
├─ infer.py            # 학습 결과 즉시 검증용 추론
└─ data/
   ├─ dummy.jsonl      # 더미 6쌍 (인프라 검증용)
   └─ schema.md        # 본 학습 데이터 스키마
```

## 1. 환경 셋업 (둘 중 하나 선택)

### A. conda
```bash
conda env create -f environment.yml
conda activate longstone-llm-ft
```

### B. pip + venv
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

> **CUDA wheel 주의**: `--extra-index-url`을 빼면 PyPI 기본 wheel(CPU only)이 받아져서 학습 시 CUDA 미발견 에러가 납니다.

## 2. OS별 함정

| OS | 추가 조치 | 이유 |
| --- | --- | --- |
| **Windows** | 모든 학습/추론 명령에 `PYTHONUTF8=1` prefix (또는 `python -X utf8 ...`) | trl이 jinja 템플릿(deepseekv3.jinja 등)을 인코딩 미명시로 read → cp949 환경에서 UnicodeDecodeError |
| **Linux** | 추가 조치 없음 | 기본 UTF-8 |

PowerShell에서 한 번에 환경변수 세팅:
```powershell
$env:PYTHONUTF8="1"
```

## 3. HuggingFace 사전 작업

```bash
huggingface-cli login   # 사용자 토큰 입력
```

**Gemma 라이선스 동의**: 토큰만으로는 weight download가 403입니다. 브라우저로 [`huggingface.co/google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) 접속 → "Acknowledge license" 클릭 (계정당 1회).

## 4. 환경 점검

```bash
python env_check.py
```

확인 항목:
1. PyTorch / CUDA / GPU 이름 / VRAM 총량
2. HF 캐시 위치 및 디스크 잔여 (15GB 미만이면 경고)
3. Gemma-2-2b-it 4-bit 로딩 smoke test (peak VRAM 출력)

`env_check.py` 출력 전체를 채민에게 공유 부탁드립니다 — 이후 명령 권장값이 GPU 사양에 따라 달라집니다.

## 5. 데이터

- **현재 동봉 (`data/dummy.jsonl`, 6쌍)**: 인프라 dry run 전용. 자세 3종(The_Seal / Spine_Stretch / Bridging) × 케이스 2건씩.
- **본 학습용 `data/sft.jsonl` (300쌍 목표)**: 채민과 별도 합의 후 전달 예정. 스키마는 [`data/schema.md`](data/schema.md) 참조 (Alpaca instruction/input/output, 한국어, 2문장 이내).

> 우선 dummy로 환경 + 학습 파이프라인 통과 확인까지 가시면 본 데이터 수신 후 바로 본 학습으로 넘어갈 수 있습니다.

## 6. 세 모델 full FT 명령

`train_full.py`의 기본값은 사용자 데스크탑(RTX 3050 8GB) 기준 — `--batch-size 1 --grad-accum 8 --max-seq 384 --epochs 1 --lr 2e-5`. 연구실 GPU 사양에 맞게 상향(§7) 권장.

```bash
# (Windows면 각 명령 앞에 PYTHONUTF8=1 prefix)

# A. Gemma-2-2b-it (사용자 환경: OOM peak 14.5GB / 24GB+ GPU 권장)
python train_full.py \
  --base-model google/gemma-2-2b-it \
  --data data/dummy.jsonl \
  --output checkpoints/full_gemma

# B. Qwen2.5-0.5B-Instruct (사용자 환경 성공, loss 3.29)
python train_full.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/dummy.jsonl \
  --output checkpoints/full_qwen05

# C. HyperCLOVAX-SEED-Text-Instruct-0.5B (사용자 환경 성공, loss 2.89, 한국어 가장 매끄러움)
python train_full.py \
  --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --data data/dummy.jsonl \
  --output checkpoints/full_hcx05
```

OOM 발생 시 `train_full.py`가 자동으로 `reports/full_<model_slug>_oom.md`를 생성합니다 (peak VRAM, 설정, 트레이스 포함). 이 파일이 발표 자료의 핵심 증거가 됩니다.

## 7. 하이퍼파라미터 상향 가이드

| GPU VRAM | 권장 시작값 |
| --- | --- |
| 8 GB | `--batch-size 1 --grad-accum 8 --max-seq 384` (기본) |
| 16 GB | `--batch-size 2 --grad-accum 4 --max-seq 512` |
| 24 GB+ | `--batch-size 4 --grad-accum 2 --max-seq 512` |
| 40 GB+ | `--batch-size 8 --grad-accum 1 --max-seq 1024` |

본 학습 시 epoch는 3~5 권장 (`--epochs 3`). 학습률(`--lr`)은 full FT 특성상 작게(2e-5) 유지 권장.

## 8. 학습 결과 즉시 확인

```bash
python infer.py \
  --base-model checkpoints/full_<slug> \
  --input-file data/dummy.jsonl \
  --n 3
```

각 샘플마다 응답시간(ms), peak VRAM(MiB), 입력, 모델 출력이 표시됩니다.

## 9. 결과 회수

- **반드시 회수**: 학습 종료 시 콘솔 last-loss, `reports/full_<slug>_oom.md` (있다면), `infer.py` 출력 일부
- **추후 합의**: checkpoint 디렉토리 자체(0.5B≈1GB, 2B≈5GB) — Drive 업로드 / 외장 SSD / 별도 협의

git push로 코드/로그 정도는 즉시 공유 가능, 모델 weight은 `.gitignore` 처리 권장.

## 10. 참고: 사용자 데스크탑(RTX 3050 8GB) 검증 결과

| 모델 | 결과 | 비고 |
| --- | --- | --- |
| Gemma-2-2b-it (full FT) | ❌ OOM peak **14.5GB** vs VRAM 8GB | 모든 메모리 절약 옵션(bf16 + gradient checkpointing + 8-bit Adam) 적용에도 불가 |
| Qwen2.5-0.5B (full FT) | ✅ loss 3.29 | dummy 6쌍 1 epoch |
| HCX-SEED-0.5B (full FT) | ✅ loss **2.89** | dummy 6쌍 1 epoch, 한국어 출력 가장 매끄러움 |
| Gemma-2-2b QLoRA (별도 트랙) | ✅ loss 3.31, 어댑터 25MB | 본 핸드오프 범위 밖 |

연구실에서 **Gemma-2-2b full FT가 성공**하면 교수님 피드백("full FT 우선, 정 안 되면 LoRA")의 정공법 증거가 확보됩니다.

## 11. 막히면

`env_check.py` 전체 출력 + 실패한 명령 + 에러 트레이스를 채민에게 공유 부탁드립니다. 의존성 페어가 까다로워(특히 `bitsandbytes` / `pyarrow`) 환경 단계에서 막히는 경우가 가장 많습니다.
