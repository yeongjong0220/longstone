import os
import sys

# [필수] 윈도우 환경 메모리 및 라이브러리 충돌 방지 설정
os.environ["HF_HUB_DISABLE_MMAP"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
import huggingface_hub
huggingface_hub.utils.disable_progress_bars()

import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("="*50)
print("🚀 [시스템 시작] OnPose 실시간 라이브 코칭 (웹캠 + LLM)")
print("="*50)

# ==========================================
# 1. 모델 설정 및 로드 (4-bit 양자화)
# ==========================================
model_id = "google/gemma-2-2b-it" 

print("⏳ 로컬 메모리에 Gemma 모델을 적재하는 중입니다...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        use_safetensors=False # 윈도우 mmap 버그 우회
    )
    print("✅ sLLM 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 오류: {e}")
    print("💡 팁: 'pip install --upgrade bitsandbytes accelerate transformers' 를 터미널에 입력해 업데이트해 보세요.")
    sys.exit()

# ==========================================
# 2. MediaPipe 초기화 및 기준값 설정
# ==========================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 'The Seal' 구르기 동작의 이상적인 기준 각도 (도)
REF_HIP = 80.0
REF_KNEE = 35.0
TOLERANCE = 15.0 # 허용 오차

def calculate_angle(a, b, c):
    """세 점(a, b, c)을 이용해 3D 공간 각도를 계산합니다."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 3. 실시간 웹캠 분석 루프
# ==========================================
cap = cv2.VideoCapture(0)
error_logs = {"hip": [], "knee": []}

print("\n" + "*"*50)
print("🎥 실시간 자세 분석 시작")
print("- 웹캠 앞에서 'The Seal' 동작을 수행하세요.")
print("- 'f' 키: 누적된 오류 분석 후 AI 피드백 생성")
print("- 'q' 키: 프로그램 종료")
print("*"*50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 거울 모드 및 MediaPipe 처리
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        try:
            # 주요 관절 좌표 추출 (왼쪽 기준)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z]

            # 실시간 각도 계산
            cur_hip = calculate_angle(shoulder, hip, knee)
            cur_knee = calculate_angle(hip, knee, ankle)
            
            status = "[PASS]"
            color = (0, 255, 0) # 녹색
            
            # 오차 범위 15도를 넘어가면 에러 로그 수집
            if abs(cur_hip - REF_HIP) > TOLERANCE or abs(cur_knee - REF_KNEE) > TOLERANCE:
                status = "[WARNING]"
                color = (0, 0, 255) # 빨간색
                error_logs["hip"].append(cur_hip)
                error_logs["knee"].append(cur_knee)

            # 화면에 뼈대 그리기 및 정보 출력
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Hip: {int(cur_hip)} (Ref: {int(REF_HIP)})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Knee: {int(cur_knee)} (Ref: {int(REF_KNEE)})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Error Frames: {len(error_logs['hip'])}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        except: pass

    cv2.imshow('On-device Pilates Coach', image)
    key = cv2.waitKey(1) & 0xFF
    
    # ==========================================
    # 4. 'f' 키 입력 시 LLM 추론 (피드백 생성)
    # ==========================================
    if key == ord('f'):
        error_frames = len(error_logs["hip"])
        if error_frames < 5:
            print("💡 충분한 오류 데이터가 모이지 않았습니다. 자세 이탈 후 다시 시도해주세요.")
            continue
            
        print("\n🧠 AI 강사가 방금 수행한 동작을 분석하고 있습니다... (잠시만 기다려주세요)\n")
        
        # 에러 통계 요약 (평균값 계산)
        avg_err_hip = int(np.mean(error_logs["hip"]))
        avg_err_knee = int(np.mean(error_logs["knee"]))
        
        hip_state = "굽혀" if avg_err_hip < REF_HIP else "펴"
        knee_state = "굽혀" if avg_err_knee < REF_KNEE else "펴"

        # 실시간 데이터를 바탕으로 동적 프롬프트 생성
        llm_prompt = f"""
        시스템: 사용자가 'The Seal' 필라테스 동작을 수행했습니다.
        분석 결과: 구르는 동작 중 총 {error_frames}프레임 동안 자세 이탈이 감지되었습니다.
        상세 내용: 기준 고관절 각도는 {int(REF_HIP)}도이나 평균 {avg_err_hip}도로 너무 {hip_state}졌고, 무릎 각도는 {int(REF_KNEE)}도이나 평균 {avg_err_knee}도로 너무 {knee_state}졌습니다.
        명령: 이 데이터를 바탕으로 사용자에게 친절하고 구체적인 자세 교정 피드백을 2문장 이내로 작성해주세요.
        """

        try:
            # Chat Template 적용
            chat = [{"role": "user", "content": llm_prompt}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=100, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 답변 디코딩
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            print("==================================================")
            print("🎙️ [AI 트레이너 피드백]")
            print(response.strip())
            print("==================================================")
            
            # 분석 완료 후 다음 분석을 위해 에러 로그 초기화
            error_logs = {"hip": [], "knee": []}
            
        except Exception as e:
            print(f"❌ 피드백 생성 중 오류 발생: {e}")

    # 'q' 키 입력 시 종료
    elif key == ord('q'):
        print("👋 프로그램을 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()