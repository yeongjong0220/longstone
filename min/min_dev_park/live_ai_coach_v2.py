import os
import sys
import collections
import pandas as pd
from scipy.signal import savgol_filter

# [필수] 윈도우 환경 메모리 및 라이브러리 충돌 방지 설정
os.environ["HF_HUB_DISABLE_MMAP"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("="*50)
print("🚀 [시스템 시작] OnPose 실시간 라이브 코칭 (웹캠 + LLM + 실시간 전처리 적용)")
print("="*50)

# ==========================================
# 1. 모델 설정 및 로드 (4-bit 양자화)
# ==========================================
model_id = "google/gemma-2-2b-it" 

print("⏳ 로컬 메모리에 Gemma 모델을 적재하는 중입니다...")
# 안정성을 위해 float16 로드로 변경하며, CPU 메모리가 꽉차서 튕기는 현상(OOM)을 막기 위해 그래픽카드로만 로드합니다.
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"  # auto 대신 cuda로 고정하여 CPU RAM 초과 방지
    )
    print("✅ sLLM 로드 완료!")
except Exception as e:
    import traceback
    print(f"❌ 모델 로드 오류: {e}")
    traceback.print_exc()
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
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# [중요 변경 부분] 전처리: 정규화 & 평활화 모듈 
# ==========================================
def process_coordinate(series, window=15, poly_order=3, threshold=3):
    """01_normalize_02.py 의 스무딩 로직 적용"""
    # 1. 이상치 처리 (Rolling Median 기반)
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
    difference = np.abs(series - rolling_med)
    median_abs_deviation = difference.rolling(window=window, center=True, min_periods=1).median()
    
    outlier_idx = difference > (threshold * median_abs_deviation + 1e-5)
    
    series_clean = series.copy()
    series_clean[outlier_idx] = rolling_med[outlier_idx]
    
    # 결측치 보간
    series_clean = series_clean.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    # 2. 사비츠키-골레이 필터 적용 (부드러운 곡선)
    window_length = window if window % 2 != 0 else window + 1
    if len(series_clean) > window_length:
        smoothed = savgol_filter(series_clean, window_length=window_length, polyorder=poly_order)
    else:
        smoothed = series_clean.values
        
    return smoothed

buffer_window = 15
pose_buffer = collections.deque(maxlen=buffer_window)

# ==========================================
# 3. 실시간 웹캠 분석 루프
# ==========================================
cap = cv2.VideoCapture(0)
error_logs = {"hip": [], "knee": []}

print("\n" + "*"*50)
print("🎥 실시간 자세 분석 시작 (전처리 필터 적용됨)")
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
            coords = {
                'Shoulder_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                'Shoulder_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                'Shoulder_z': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z,
                'Hip_x': landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                'Hip_y': landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                'Hip_z': landmarks[mp_pose.PoseLandmark.LEFT_HIP].z,
                'Knee_x': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                'Knee_y': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                'Knee_z': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z,
                'Ankle_x': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                'Ankle_y': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                'Ankle_z': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z,
            }
            
            pose_buffer.append(coords)
            
            # 버퍼 데이터를 활용한 전처리 로직 실행
            df_buffer = pd.DataFrame(list(pose_buffer))
            
            # [전처리 1] 스케일 정규화 (Normalization) (거리값 이용) 
            # 01_normalize_02 와 비슷하게 Shoulder-Hip 기준으로 대체 적용
            dist = np.sqrt(
                (df_buffer['Shoulder_x'] - df_buffer['Hip_x'])**2 +
                (df_buffer['Shoulder_y'] - df_buffer['Hip_y'])**2 +
                (df_buffer['Shoulder_z'] - df_buffer['Hip_z'])**2
            )
            mean_len = dist.mean()
            
            coord_cols = df_buffer.columns
            if mean_len > 0:
                df_buffer[coord_cols] = df_buffer[coord_cols] / mean_len
                
            # [전처리 2] 부드러운 스무딩 (Smoothing & Clean)
            for col in coord_cols:
                df_buffer[col] = process_coordinate(df_buffer[col], window=buffer_window, poly_order=3, threshold=3)
                
            # 가장 최신 전처리된 프레임 좌표를 가져옴
            last_frame = df_buffer.iloc[-1]
            
            shoulder_sm = [last_frame['Shoulder_x'], last_frame['Shoulder_y'], last_frame['Shoulder_z']]
            hip_sm = [last_frame['Hip_x'], last_frame['Hip_y'], last_frame['Hip_z']]
            knee_sm = [last_frame['Knee_x'], last_frame['Knee_y'], last_frame['Knee_z']]
            ankle_sm = [last_frame['Ankle_x'], last_frame['Ankle_y'], last_frame['Ankle_z']]

            # 실시간 각도 계산 (전제 처리된 벡터를 이용하므로 지터 방축 효과)
            cur_hip = calculate_angle(shoulder_sm, hip_sm, knee_sm)
            cur_knee = calculate_angle(hip_sm, knee_sm, ankle_sm)
            
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
            
        except Exception as e:
            # 버퍼가 부족하거나 에러 발생 시 건너뜀
            pass

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
