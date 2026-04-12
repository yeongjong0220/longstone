import os
import sys
import collections
import urllib.request
import json
import pandas as pd
from scipy.signal import savgol_filter
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# [필수 1] Gemini REST API 설정 (무료)
# ==========================================
GOOGLE_API_KEY = "AIzaSyDR0vlu1JMkDfwG93opful7EiRmnfteY9Q"

print("="*50)
print("🚀 [시스템 시작] OnPose 실시간 라이브 코칭 (웹캠 + Gemini REST API)")
print("✅ 최신 MediaPipe Tasks API 적용 (완벽 호환)")
print("="*50)

# ==========================================
# 2. MediaPipe 최신 Tasks API 초기화
# ==========================================
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("⏳ 최신 뼈대 인식 모델(Pose Landmarker)을 다운로드 중입니다... (최초 1회, 약 3MB)")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("✅ 다운로드 완료!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# MediaPipe 기존 그리기 도구 (Tasks API용으로 래핑)
mp_drawing = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None

REF_HIP = 80.0
REF_KNEE = 35.0
TOLERANCE = 15.0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def process_coordinate(series, window=15, poly_order=3, threshold=3):
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
    difference = np.abs(series - rolling_med)
    median_abs_deviation = difference.rolling(window=window, center=True, min_periods=1).median()
    
    outlier_idx = difference > (threshold * median_abs_deviation + 1e-5)
    
    series_clean = series.copy()
    series_clean[outlier_idx] = rolling_med[outlier_idx]
    series_clean = series_clean.interpolate().bfill().ffill()
    
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
print("🎥 실시간 자세 분석 시작")
print("- 웹캠 앞에서 'The Seal' 동작을 수행하세요.")
print("- 'f' 키: 누적된 오류 분석 후 AI 피드백 생성 (초고속)")
print("- 'q' 키: 프로그램 종료")
print("*"*50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 태스크 형식으로 변환 후 감지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = detector.detect(mp_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(results.pose_landmarks) > 0:
        landmarks = results.pose_landmarks[0] # 첫 번째 사람
        try:
            # 주요 관절 좌표 추출 (11: 좌측 어깨, 23: 좌측 골반, 25: 좌측 무릎, 27: 좌측 발목)
            coords = {
                'Shoulder_x': landmarks[11].x, 'Shoulder_y': landmarks[11].y, 'Shoulder_z': landmarks[11].z,
                'Hip_x': landmarks[23].x, 'Hip_y': landmarks[23].y, 'Hip_z': landmarks[23].z,
                'Knee_x': landmarks[25].x, 'Knee_y': landmarks[25].y, 'Knee_z': landmarks[25].z,
                'Ankle_x': landmarks[27].x, 'Ankle_y': landmarks[27].y, 'Ankle_z': landmarks[27].z,
            }
            
            pose_buffer.append(coords)
            df_buffer = pd.DataFrame(list(pose_buffer))
            
            dist = np.sqrt(
                (df_buffer['Shoulder_x'] - df_buffer['Hip_x'])**2 +
                (df_buffer['Shoulder_y'] - df_buffer['Hip_y'])**2 +
                (df_buffer['Shoulder_z'] - df_buffer['Hip_z'])**2
            )
            mean_len = dist.mean()
            coord_cols = df_buffer.columns
            if mean_len > 0:
                df_buffer[coord_cols] = df_buffer[coord_cols] / mean_len
                
            for col in coord_cols:
                df_buffer[col] = process_coordinate(df_buffer[col], window=buffer_window)
                
            last_frame = df_buffer.iloc[-1]
            shoulder_sm = [last_frame['Shoulder_x'], last_frame['Shoulder_y'], last_frame['Shoulder_z']]
            hip_sm = [last_frame['Hip_x'], last_frame['Hip_y'], last_frame['Hip_z']]
            knee_sm = [last_frame['Knee_x'], last_frame['Knee_y'], last_frame['Knee_z']]
            ankle_sm = [last_frame['Ankle_x'], last_frame['Ankle_y'], last_frame['Ankle_z']]

            cur_hip = calculate_angle(shoulder_sm, hip_sm, knee_sm)
            cur_knee = calculate_angle(hip_sm, knee_sm, ankle_sm)
            
            status, color = "[PASS]", (0, 255, 0)
            
            if abs(cur_hip - REF_HIP) > TOLERANCE or abs(cur_knee - REF_KNEE) > TOLERANCE:
                status, color = "[WARNING]", (0, 0, 255)
                error_logs["hip"].append(cur_hip)
                error_logs["knee"].append(cur_knee)

            # 새 버전에서는 커스텀 뼈대 그리기 로직 (점 하나씩 찍기 - solutions 모듈 부재 대비)
            for lm in [landmarks[11], landmarks[23], landmarks[25], landmarks[27]]:
                cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(image, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Hip: {int(cur_hip)} (Ref: {int(REF_HIP)})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Knee: {int(cur_knee)} (Ref: {int(REF_KNEE)})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Error Frames: {len(error_logs['hip'])}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        except Exception as e:
            pass

    cv2.imshow('On-device Pilates Coach (API)', image)
    key = cv2.waitKey(1) & 0xFF
    
    # ==========================================
    # 4. 'f' 키 입력 시 API 호출 (피드백 생성)
    # ==========================================
    if key == ord('f'):
        error_frames = len(error_logs["hip"])
        if error_frames < 5:
            print("💡 충분한 오류 데이터가 모이지 않았습니다. (Error Frames 5 이상 필요)")
            continue
            
        print("\n🧠 구글 클라우드 AI가 동작을 분석 중입니다... (1~2초 소요)\n")
        
        avg_err_hip = int(np.mean(error_logs["hip"]))
        avg_err_knee = int(np.mean(error_logs["knee"]))
        hip_state = "굽혀" if avg_err_hip < REF_HIP else "펴"
        knee_state = "굽혀" if avg_err_knee < REF_KNEE else "펴"

        # 프롬프트 구성
        llm_prompt = f"""
        당신은 친절하고 전문적인 필라테스 강사입니다. 사용자가 'The Seal' 동작을 수행했습니다.
        분석 결과: 구르는 동작 중 총 {error_frames}프레임 동안 자세 이탈이 감지되었습니다.
        상세 내용: 기준 고관절 각도는 {int(REF_HIP)}도이나 평균 {avg_err_hip}도로 너무 {hip_state}졌고, 무릎 각도는 {int(REF_KNEE)}도이나 평균 {avg_err_knee}도로 너무 {knee_state}졌습니다.
        명령: 이 데이터를 바탕으로 사용자에게 친절하고 구체적인 자세 교정 피드백을 단 2문장 이내로 직관적이게 작성해주세요.
        """

        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
            data = {"contents": [{"parts": [{"text": llm_prompt}]}]}
            req = urllib.request.Request(api_url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                feedback_text = result['candidates'][0]['content']['parts'][0]['text']

            print("==================================================")
            print("🎙️ [초고속 AI 트레이너 피드백]")
            print(feedback_text.strip())
            print("==================================================")
            
            error_logs = {"hip": [], "knee": []}
            
        except Exception as e:
            print(f"❌ API 호출 중 오류 발생: {e}")

    elif key == ord('q'):
        print("👋 프로그램을 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
