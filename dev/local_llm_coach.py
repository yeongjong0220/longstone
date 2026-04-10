import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. 모델 설정
# ==========================================
# 사용하실 모델명 (Gemma 1B 또는 2B Instruct 모델의 정확한 허깅페이스 저장소 이름)
model_id = "google/gemma-2-2b-it" 

# [핵심] Jetson Nano 메모리 최적화를 위한 4-bit 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("⏳ 로컬 메모리에 Gemma 모델을 적재하는 중입니다... (최초 실행 시 다운로드 소요)")

# 토크나이저 및 모델 로드 (디바이스 자동 할당)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# ==========================================
# 2. 앞서 상태 머신에서 추출된 상황 프롬프트
# ==========================================
llm_prompt = """
시스템: 사용자가 'The Seal' 필라테스 동작을 수행했습니다.
분석 결과: 구르는 동작 중 '고관절 과도하게 굽힘' 문제가 총 13프레임 동안 감지되었습니다.
명령: 이 데이터를 바탕으로 사용자에게 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 작성해주세요.
"""

# ==========================================
# 3. 모델 추론 (피드백 생성)
# ==========================================
print("\n🧠 AI가 맞춤형 피드백을 생성하고 있습니다...\n")

# Gemma 모델의 지시어(Instruction) 포맷에 맞게 텍스트 변환
chat = [{"role": "user", "content": llm_prompt}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# 텐서 변환 및 GPU(또는 CPU) 할당
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 텍스트 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=100, # 2문장 생성이므로 100 토큰이면 충분 (속도 최적화)
    temperature=0.7,    # 0.0(단답형) ~ 1.0(창의적). 0.7은 자연스러운 대화형 수치
    do_sample=True
)

# 입력 프롬프트 부분을 제외하고, 모델이 새롭게 생성한 '답변'만 잘라내어 디코딩
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

# ==========================================
# 4. 결과 출력
# ==========================================
print("==================================================")
print("🎙️ [AI 트레이너 피드백]")
print(response.strip())
print("==================================================")
