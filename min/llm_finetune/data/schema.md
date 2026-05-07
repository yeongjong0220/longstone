# SFT 데이터 스키마

런타임 코칭 코드가 LLM 에 보내는 입력 분포와 학습 데이터의 input 분포가 **정확히 일치**해야 어댑터가 실시간 파이프라인에 그대로 swap된다. 현재 코드 베이스에는 두 종류의 런타임 입력 분포가 공존한다:

- **v3 (`min/min_dev_park/live_ai_coach_v3_api.py:271-275`)** — 단순 평균 각도 + error_frames. `dummy.jsonl` 6쌍이 이 분포로 작성됨.
- **v4 (`live_ai_coach_v4_bundle/min/min_dev_park/live_ai_coach_v4_quality.py:261-281`)** — Mahalanobis 거리·국면별 PASS/WARN/FAIL·top_issue·worst_frames 가 포함된 풍부한 요약. **본 학습용 `data/sft.jsonl` 은 v4 분포로 합성한다.**

`data/synth/` 의 합성 파이프라인이 v4 형태를 그대로 따른다 (`live_ai_coach_v4_bundle/pilates_temporal_lifter/runtime_quality.py:124-136` 의 `format_summary_for_prompt` 함수를 직접 import해서 직렬화).

## 1. 파일 형식
- 경로: `min/llm_finetune/data/sft.jsonl` (gitignore)
- 한 줄 = 한 학습 샘플 (JSON Lines, UTF-8, LF)

## 2. 필드
| key | 타입 | 설명 |
| --- | --- | --- |
| `instruction` | str | 시스템 지시문(페르소나 + 출력 규칙). 모든 샘플에서 동일 권장. |
| `input` | str | 자세별 통계 (변수치 채워진 형태). 다중 자세 지원. |
| `output` | str | 모범 코칭 응답 (한국어, 2문장 이내). |

## 3. `instruction` 고정 문구
```
당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.
```

## 4. `input` 템플릿 (다중 자세)

**현재 지원되는 자세 (`POSE_CONFIG`)**:
| key | name_en | name_kr | ref_hip | ref_knee | tolerance |
| --- | --- | --- | --- | --- | --- |
| `The_Seal` | The Seal | 더 씰 | 80° | 35° | 15° |
| `Spine_Stretch` | Spine Stretch | 스파인 스트레치 | 80° | 175° | 15° |
| `Bridging` | Bridging | 브릿징 | 170° | 90° | 15° |

POSE_CONFIG가 늘어나면 그대로 input 변수만 채워 추가 가능.

### 4.1 채우기 변수
| 변수 | 의미 | 비고 |
| --- | --- | --- |
| `name_en` | 동작 영문명 | POSE_CONFIG[key]['name_en'] |
| `name_kr` | 동작 한글명 | POSE_CONFIG[key]['name_kr'] |
| `error_frames` | 자세 이탈 누적 프레임 수 | 5 ~ 200 |
| `ref_hip` | 자세별 기준 고관절 각도 | int(cfg['ref_hip']) |
| `avg_err_hip` | 이탈 구간 평균 고관절 각도 | int |
| `hip_state` | "굽혀" 또는 "펴" (`avg_err_hip < ref_hip` → "굽혀") | — |
| `ref_knee` | 자세별 기준 무릎 각도 | int(cfg['ref_knee']) |
| `avg_err_knee` | 이탈 구간 평균 무릎 각도 | int |
| `knee_state` | "굽혀" 또는 "펴" | — |

### 4.2 템플릿 (`live_ai_coach_v3_api.py:271-275`와 동일)
```
사용자가 '{name_en}({name_kr})' 동작을 수행했습니다. 동작 중 총 {error_frames}프레임 동안 자세 이탈이 감지되었습니다.
기준 고관절 각도는 {ref_hip}도이나 평균 {avg_err_hip}도로 너무 {hip_state}졌고, 무릎 각도는 {ref_knee}도이나 평균 {avg_err_knee}도로 너무 {knee_state}졌습니다.
```

## 5. `output` 작성 가이드
- 길이: 2문장 이내, 약 60~120자
- 한국어, 친절·구체적·존댓말
- 통계에 직접 대응하는 교정 동작을 1개 이상 언급
- 해당 자세(name_en) 특성 반영(예: Bridging은 엉덩이 들기, Spine Stretch는 척추 스트레치)
- 추측·환각 금지(예: "허리 통증" 같은 의학적 추론 X)

## 5b. v4 input 스키마 (sft.jsonl 본 분포)

`live_ai_coach_v4_quality.py` 가 매 세션 종료 시 생성하는 요약 dict 를 `format_summary_for_prompt(pose_name_en, summary)` 로 평문 직렬화한 결과를 그대로 input 필드에 넣는다.

### 5b.1 직렬화 결과 (예시)
```
Pose: The Seal
Frames analyzed: 62
PASS/WARN/FAIL counts: {'PASS': 50, 'WARN': 10, 'FAIL': 2}
Top issue features: {'hip_angle': 8, 'knee_symmetry': 4}
Mean Mahalanobis d2: 1.234
Max Mahalanobis d2: 3.456
Worst frames: [{'frame': 45, 'phase_name': 'APEX', 'status': 'WARN', 'mahalanobis_d2': 2.8, 'top_feature': 'hip_angle', 'top_feature_z': 2.1}, ...]
```

### 5b.2 자세별 phase / feature 어휘 (`runs/<run>/model.json` 그대로)
| 자세 | scorer | phase_scheme | phase_names | feature_names |
| --- | --- | --- | --- | --- |
| The Seal (더 씰) | ✅ `the_seal_mahalanobis_hip_knee_all_v2` | cyclic4 | READY, OUTBOUND, APEX, RETURN | hip_angle, knee_angle, hip_velocity, knee_velocity |
| Bridging (브릿징) | ✅ `bridging_mahalanobis_v1` | progress3 | START, MIDDLE, END | hip_angle, knee_angle, trunk_angle, hip_symmetry, knee_symmetry, hip_velocity, knee_velocity, trunk_velocity |
| Spine Stretch (스파인 스트레치) | ❌ fallback only | — | (없음) | hip, knee (단순 각도 비교) |

### 5b.3 fallback summary (Spine Stretch)
scorer 가 없는 케이스는 `live_ai_coach_v4_quality.py:307-320` 의 `fallback_summary` 가 만드는 dict 를 사용. `phase_counts` 는 빈 dict, `mean_mahalanobis_d2` / `max_mahalanobis_d2` 는 0.0, `worst_frames` 는 `{frame, feature, value, ref}` 스키마 (Mahalanobis 케이스와 키가 다르다).

### 5b.4 v4 instruction (고정)
```
당신은 정확하지만 격려하는 필라테스 강사입니다. 다음 분석 요약을 바탕으로 한국어 존댓말로 정확히 3문장 자세 교정 피드백을 작성하세요. (1) 전반적 품질, (2) top issue feature 또는 phase 기반 가장 중요한 교정점, (3) 다음 시도 시 적용할 단순 큐. 구현 디테일(Mahalanobis 등)은 언급하지 마세요.
```
모든 v4 샘플에서 동일.

### 5b.5 output 작성 가이드 (v4)
- 정확히 3문장, 80~350자.
- 한국어 존댓말. 한국어 비율 ≥ 0.6.
- input 의 phase 이름 또는 top issue feature 이름 중 적어도 하나를 자연스럽게 인용 (영문 그대로 또는 한글 의역 모두 가능. `data/synth/filter.py` 의 `FEATURE_KOR_HINTS` 표 참고).
- 자세 특성 반영: The Seal → 컴팩트한 C-curve / 모멘텀, Bridging → 어깨-골반-무릎 정렬·복부 안정, Spine Stretch → 척추 분절·햄스트링.
- 구현 디테일(Mahalanobis, 점수, threshold 등) 언급 금지. 의학적 추론 금지.

## 6. 채팅 포맷 변환 (학습 시)
`train_lora.py`는 다음과 같이 chat template 적용:

```python
user_text = f"{instruction}\n\n{input}"
chat = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": output},
]
formatted = tokenizer.apply_chat_template(chat, tokenize=False)
```

→ `live_ai_coach_v3_api.py`의 추론 프롬프트와 거의 동일한 구조.

## 7. 예시

### 7.1 v3 분포 (dummy.jsonl, 톤 가이드용 시드)

```json
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'The Seal(더 씰)' 동작을 수행했습니다. 동작 중 총 47프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 80도이나 평균 68도로 너무 펴졌고, 무릎 각도는 35도이나 평균 43도로 너무 펴졌습니다.", "output": "고관절을 조금 더 깊게 굽혀 척추 만곡을 유지해보세요. 무릎도 가슴 쪽으로 더 끌어당기면 구르는 동작이 한결 안정됩니다."}
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'Spine Stretch(스파인 스트레치)' 동작을 수행했습니다. 동작 중 총 32프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 80도이나 평균 95도로 너무 굽혀졌고, 무릎 각도는 175도이나 평균 158도로 너무 굽혀졌습니다.", "output": "다리는 무릎 뒤를 길게 밀어 곧게 펴주시고, 골반에서부터 천천히 척추를 말아 내리듯 숙여보세요. 그러면 척추 한 마디씩 늘어나는 감각이 느껴집니다."}
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'Bridging(브릿징)' 동작을 수행했습니다. 동작 중 총 70프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 170도이나 평균 150도로 너무 굽혀졌고, 무릎 각도는 90도이나 평균 110도로 너무 펴졌습니다.", "output": "엉덩이를 조금 더 위로 밀어 올려 어깨-골반-무릎이 일직선이 되도록 만들어보세요. 무릎은 발목 바로 위에 오게 살짝 모아주면 햄스트링 활성화가 더 잘 일어납니다."}
```

### 7.2 v4 분포 (sft.jsonl 본 학습, 합성 산출 형태)

```json
{"instruction": "당신은 정확하지만 격려하는 필라테스 강사입니다. 다음 분석 요약을 바탕으로 한국어 존댓말로 정확히 3문장 자세 교정 피드백을 작성하세요. (1) 전반적 품질, (2) top issue feature 또는 phase 기반 가장 중요한 교정점, (3) 다음 시도 시 적용할 단순 큐. 구현 디테일(Mahalanobis 등)은 언급하지 마세요.", "input": "Pose: The Seal\nFrames analyzed: 62\nPASS/WARN/FAIL counts: {'PASS': 50, 'WARN': 10, 'FAIL': 2}\nTop issue features: {'hip_angle': 8, 'knee_symmetry': 4}\nMean Mahalanobis d2: 1.234\nMax Mahalanobis d2: 3.456\nWorst frames: [{'frame': 45, 'phase_name': 'APEX', 'status': 'WARN', 'mahalanobis_d2': 2.8, 'top_feature': 'hip_angle', 'top_feature_z': 2.1}]", "output": "전반적으로 동작은 안정적이지만 APEX 구간에서 고관절 각도가 평소보다 크게 흔들렸습니다. 가장 시급한 교정은 가슴과 무릎이 멀어지는 순간을 줄여 고관절을 더 깊이 접는 것입니다. 다음 시도에서는 정점에서 \"가슴-무릎 거리 그대로\" 라는 큐를 떠올리며 한 박자 더 머무르듯 굴러보세요."}
```

## 8. 합성 파이프라인

`data/synth/` 의 4단계 스크립트로 v4 본 데이터셋을 만든다 — `build_inputs.py` → `generate_outputs.py` → `filter.py` → `split.py`. 자세한 절차는 [../CLAUDE.md](../CLAUDE.md) 의 합성 절차 섹션 참고.
