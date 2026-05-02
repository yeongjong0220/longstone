# SFT 데이터 스키마

`min/min_dev_park/live_ai_coach_v3_api.py`(wonpark develop 버전)의 `request_feedback()` 함수가 LLM에 보내는 동적 프롬프트와 **분포가 일치**하도록 정의한다. 학습-추론 분포를 맞춰야 어댑터가 실시간 파이프라인에 그대로 swap된다.

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

## 7. 예시 (3가지 자세 각 1건)

```json
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'The Seal(더 씰)' 동작을 수행했습니다. 동작 중 총 47프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 80도이나 평균 68도로 너무 펴졌고, 무릎 각도는 35도이나 평균 43도로 너무 펴졌습니다.", "output": "고관절을 조금 더 깊게 굽혀 척추 만곡을 유지해보세요. 무릎도 가슴 쪽으로 더 끌어당기면 구르는 동작이 한결 안정됩니다."}
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'Spine Stretch(스파인 스트레치)' 동작을 수행했습니다. 동작 중 총 32프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 80도이나 평균 95도로 너무 굽혀졌고, 무릎 각도는 175도이나 평균 158도로 너무 굽혀졌습니다.", "output": "다리는 무릎 뒤를 길게 밀어 곧게 펴주시고, 골반에서부터 천천히 척추를 말아 내리듯 숙여보세요. 그러면 척추 한 마디씩 늘어나는 감각이 느껴집니다."}
{"instruction": "당신은 친절하고 전문적인 필라테스 강사입니다. 다음 분석 통계를 보고 해당 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 직관적이게 작성하세요.", "input": "사용자가 'Bridging(브릿징)' 동작을 수행했습니다. 동작 중 총 70프레임 동안 자세 이탈이 감지되었습니다.\n기준 고관절 각도는 170도이나 평균 150도로 너무 굽혀졌고, 무릎 각도는 90도이나 평균 110도로 너무 펴졌습니다.", "output": "엉덩이를 조금 더 위로 밀어 올려 어깨-골반-무릎이 일직선이 되도록 만들어보세요. 무릎은 발목 바로 위에 오게 살짝 모아주면 햄스트링 활성화가 더 잘 일어납니다."}
```
