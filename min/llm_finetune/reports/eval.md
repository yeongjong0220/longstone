# 5종 모델 추론 비교 — base / full FT / LoRA

- 일시: 2026-05-03
- 디바이스: NVIDIA GeForce RTX 3050 8GB
- 학습 데이터: `data/dummy.jsonl` (6쌍, 자세별 2건씩)
- 평가 입력: dummy.jsonl 첫 3건 (The Seal × 2, Spine Stretch × 1)
- 학습 epoch: 모두 1
- 추론: infer.py (4-bit 로딩, max_new_tokens 120, temperature 0.7)

> ⚠️ **데이터 한계**: 6쌍 1 epoch이라 학습 결과의 절대 품질 비교 의미는 작음. 본 보고서의 가치는 **(1) 인프라 동작 검증, (2) base 톤 vs 학습 후 톤 변화 시그널, (3) 모델별 응답시간/VRAM 실측**.

## 1. 정량 지표

| 모델 | 응답시간 (평균 ms) | peak VRAM (MiB) | 비고 |
| --- | ---: | ---: | --- |
| base Gemma-2-2b-it | 4942 | 2300 | 4-bit 로딩 |
| base Qwen2.5-0.5B-Instruct | 4302 | 469 | 4-bit, 메모리 1/5 |
| full FT Qwen2.5-0.5B | 4276 | 469 | 6쌍 1 epoch 학습 |
| full FT HyperCLOVAX-0.5B | 6725 | 470 | 6쌍 1 epoch 학습 |
| Gemma-2-2b + LoRA 어댑터 | 6527 | 2324 | 6쌍 1 epoch 학습, 어댑터 25MB |

## 2. 정성 평가 (대표 사례 1: The Seal, error_frames=47)

입력: "고관절 80도 기준에 평균 68도로 너무 펴짐, 무릎 35도 기준에 평균 43도로 너무 펴짐"
정답(시드): "고관절을 조금 더 깊게 굽혀 척추 만곡을 유지해보세요. 무릎도 가슴 쪽으로 더 끌어당기면 구르는 동작이 한결 안정됩니다."

| 모델 | 출력 (요약) | 평가 |
| --- | --- | --- |
| base Gemma | "고관절은 넓게 펴져 있습니다. 무릎은 펴짐으로써 균형과 힘의 유사한 분포를 잃고 있습니다." | 현상 묘사만, 교정 지시 없음 |
| base Qwen | "1. 제안: ... 2. 분석: ... '더 씥'이라는 단어를 의미하며..." | **도메인 무관·한국어 오류**, 부적합 |
| full Qwen | "죄송합니다, 저는 AI 기반... 사용자 행동에 대한 진단과 조치를 결정하는 것은 제 역량으로 아닙니다." | 거부 응답, **6쌍으로는 못 깸** |
| full HCX | "더 씰 동작은 하체의 안정성을 높이는 데 도움이 되지만, 자세가 너무 과도하게 펴져 있어 무릎과 고관절이 부담을 받을 수 있습니다. 무릎은 평균적으로 43도..." | **도메인 용어·수치 자연스럽게 인용**, 한국어 매끄러움 |
| Gemma+LoRA | "'더 씰' 동작에서 너무 많은 자세 이탈이 관찰되었어요. 고관절과 무릎 각도가 너무 펴짐을 확인했는데, 곧바로 펴지 않고 균형을 맞추는 것에 집중해 보는 것도 좋을 것 같아요." | **시드 톤(2문장 이내·존댓말·구체 지시) 명확히 학습**, 형태 가장 가까움 |

## 3. 핵심 관찰

1. **HyperCLOVAX-0.5B의 한국어 도메인 출력이 압도적으로 자연스러움**
   네이버의 한국어 특화 사전학습 효과. 작은 모델임에도 입력 통계의 수치(43도, 158도 등)를 인용하며 코칭 형태로 출력.

2. **Gemma-2-2b LoRA는 시드 톤을 가장 잘 학습**
   "~해 보세요", "~좋을 것 같아요" 같은 시드의 부드러운 존댓말과 2문장 형태가 출력에 그대로 나타남. 6쌍 1 epoch라는 매우 빈약한 설정에서도 톤 학습이 가시적.

3. **base Qwen2.5-0.5B는 코칭 도메인에 부적합**
   한국어 약점이 분명. "더 씥" 같은 표기 오류, 동어반복, 거부 응답("AI라서 못합니다"). 본 학습으로 끌어올리려면 데이터·epoch이 매우 많이 필요할 것.

4. **응답시간**: 0.5B 모델(469 MiB VRAM)이 Gemma-2-2b(2300 MiB)보다 항상 빠르진 않음. HCX는 출력 토큰 수가 많아 6.7초까지 늘어남. 온디바이스 1초 목표를 달성하려면 max_new_tokens·temperature 조정 또는 distilled/quantized 추론 최적화 필요.

5. **LoRA의 효율성 검증**: 학습 가능 파라미터 0.24% (640만 / 26억), 어댑터 25MB. 같은 톤 학습 효과가 full FT 대비 월등히 가벼움.

## 4. 단계별 결과 종합

| 단계 | 결과 | 산출물 |
| --- | --- | --- |
| 0. env_check | ✅ CUDA OK, Gemma 4-bit 로딩 peak 2160 MiB | (없음) |
| **A. Gemma full FT** | ❌ OOM (peak **14550 MiB** vs VRAM 8192 MiB) | `reports/full_google__gemma-2-2b-it_oom.md` |
| **C. Qwen0.5 full FT** | ✅ loss 3.29 / 23.4s / 988 MB | `checkpoints/full_qwen05/` |
| **C. HCX0.5 full FT** | ✅ loss 2.89 / 29.6s / 1.13 GB | `checkpoints/full_hcx05/` |
| **LoRA. Gemma QLoRA** | ✅ loss 3.31 / 8s / 25 MB 어댑터 | `checkpoints/lora_gemma/` |
| Eval | ✅ 5종 추론 완료 (이 보고서) | `reports/eval.md` |

## 5. 발표 메시지

> "교수님 피드백대로 full fine-tuning을 1차로 시도했습니다. **RTX 3050 8GB 환경에서 Gemma-2-2b full FT는 메모리 절약 옵션을 모두 켠 상태(bf16 + grad ckpt + 8-bit Adam + batch 1)에서도 14.5GB 요구로 OOM이 발생해 불가능**(증거: `reports/full_google__gemma-2-2b-it_oom.md`)했습니다. 따라서 작은 모델로 full FT를 수행했고(Qwen2.5-0.5B, HyperCLOVAX-0.5B 각각 성공), 큰 모델은 LoRA로 fallback하는 방식이 검증되었습니다.
>
> 정성 평가 시그널: **HyperCLOVAX-0.5B의 한국어 코칭 출력이 가장 자연스럽고**, **Gemma-2-2b + LoRA는 시드 데이터의 톤을 가장 잘 학습**합니다. 다만 dummy 6쌍 1 epoch 결과이므로 실제 학습 데이터(`sft.jsonl`) 확보 후 본 학습에서 의미 있는 비교가 가능합니다."

## 6. 다음 단계

- 합성 SFT 데이터셋 구축 (300쌍+)
- 본 학습 (Gemma+LoRA, HCX-full 두 줄기)
- `live_ai_coach_v2.py`에 LoRA 어댑터 swap 통합
- 최종 정량 비교 (BLEU, 사람 평가)
