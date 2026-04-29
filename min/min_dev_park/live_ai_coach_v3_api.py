import os
import sys
import time
import collections
import urllib.request
import urllib.error
import json
import pandas as pd
from scipy.signal import savgol_filter
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dotenv import load_dotenv

# 스크립트 파일 위치 기준 절대 경로 (실행 위치와 무관하게 동작)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  # longstone/

# ==========================================
# [필수 1] Gemini REST API 설정 (무료)
# ==========================================
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(ENV_PATH)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PASTE_YOUR_KEY_HERE":
    raise RuntimeError(
        f"GOOGLE_API_KEY가 설정되지 않았습니다. "
        f"{ENV_PATH} 파일에 GOOGLE_API_KEY=발급받은_키 형식으로 입력하세요."
    )

print("="*50)
print("🚀 [시스템 시작] OnPose 실시간 라이브 코칭 (웹캠 + Gemini REST API)")
print("✅ 최신 MediaPipe Tasks API 적용 (완벽 호환)")
print("="*50)

# ==========================================
# 2. MediaPipe 최신 Tasks API 초기화
# ==========================================
model_path = os.path.join(PROJECT_ROOT, 'pose_landmarker_lite.task')
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

# 자세별 기준 각도 (단위: 도). Spine Stretch 값은 임시 추정치이며 골든 스탠다드 데이터로 튜닝 필요.
POSE_CONFIG = {
    "The_Seal": {
        "ref_hip": 80.0, "ref_knee": 35.0, "tolerance": 15.0,
        "name_kr": "더 씰", "name_en": "The Seal",
    },
    "Spine_Stretch": {
        "ref_hip": 80.0, "ref_knee": 175.0, "tolerance": 15.0,
        "name_kr": "스파인 스트레치", "name_en": "Spine Stretch",
    },
}

def classify_pose(knee_angle):
    # 무릎 각도로 두 자세 구분: The Seal은 무릎 깊게 굽힘, Spine Stretch는 거의 폄
    if knee_angle < 90:
        return "The_Seal"
    if knee_angle > 150:
        return "Spine_Stretch"
    return None

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
error_logs = {pose: {"hip": [], "knee": []} for pose in POSE_CONFIG}
pose_label_buffer = collections.deque(maxlen=10)  # 자세 라벨 안정화용 (다수결)
current_pose = None

print("\n" + "*"*50)
print("🎥 실시간 자세 분석 시작 (자세 자동 분류)")
print("- 'The Seal' 또는 'Spine Stretch' 동작을 수행하세요.")
print("- 코드가 무릎 각도를 보고 자동으로 어떤 자세인지 판별합니다.")
print("- 'f' 키: 현재 자세에 누적된 오류 기반 AI 피드백 생성")
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

            # 자세 자동 분류 (최근 10프레임 다수결로 안정화)
            detected = classify_pose(cur_knee)
            if detected:
                pose_label_buffer.append(detected)
            if pose_label_buffer:
                current_pose = max(set(pose_label_buffer), key=pose_label_buffer.count)

            if current_pose is None:
                status, color = "[Detecting...]", (200, 200, 200)
                ref_hip_disp, ref_knee_disp, err_count = 0, 0, 0
                pose_name = "Detecting..."
            else:
                cfg = POSE_CONFIG[current_pose]
                ref_hip_disp, ref_knee_disp = cfg["ref_hip"], cfg["ref_knee"]
                pose_name = cfg["name_en"]
                status, color = "[PASS]", (0, 255, 0)
                if abs(cur_hip - cfg["ref_hip"]) > cfg["tolerance"] or abs(cur_knee - cfg["ref_knee"]) > cfg["tolerance"]:
                    status, color = "[WARNING]", (0, 0, 255)
                    error_logs[current_pose]["hip"].append(cur_hip)
                    error_logs[current_pose]["knee"].append(cur_knee)
                err_count = len(error_logs[current_pose]["hip"])

            # 관절 점 그리기 (좌측 어깨/골반/무릎/발목)
            for lm in [landmarks[11], landmarks[23], landmarks[25], landmarks[27]]:
                cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(image, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(image, f"Status: {status}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(image, f"Hip: {int(cur_hip)} (Ref: {int(ref_hip_disp)})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Knee: {int(cur_knee)} (Ref: {int(ref_knee_disp)})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Error Frames: {err_count}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        except Exception as e:
            pass

    cv2.imshow('On-device Pilates Coach (API)', image)
    key = cv2.waitKey(1) & 0xFF
    
    # ==========================================
    # 4. 'f' 키 입력 시 API 호출 (피드백 생성)
    # ==========================================
    if key == ord('f'):
        if not current_pose:
            print("💡 자세가 아직 인식되지 않았습니다. The Seal 또는 Spine Stretch 자세를 취해주세요.")
            continue

        cfg = POSE_CONFIG[current_pose]
        logs = error_logs[current_pose]
        error_frames = len(logs["hip"])
        if error_frames < 5:
            print(f"💡 [{cfg['name_en']}] 충분한 오류 데이터가 모이지 않았습니다. ({error_frames}/5)")
            continue

        print(f"\n🧠 [{cfg['name_en']}] 동작을 구글 클라우드 AI가 분석 중입니다... (1~2초 소요)\n")

        avg_err_hip = int(np.mean(logs["hip"]))
        avg_err_knee = int(np.mean(logs["knee"]))
        hip_state = "굽혀" if avg_err_hip < cfg["ref_hip"] else "펴"
        knee_state = "굽혀" if avg_err_knee < cfg["ref_knee"] else "펴"

        # 프롬프트 구성 (현재 자세에 맞춤)
        llm_prompt = f"""
        당신은 친절하고 전문적인 필라테스 강사입니다. 사용자가 '{cfg['name_en']}({cfg['name_kr']})' 동작을 수행했습니다.
        분석 결과: 동작 중 총 {error_frames}프레임 동안 자세 이탈이 감지되었습니다.
        상세 내용: 기준 고관절 각도는 {int(cfg['ref_hip'])}도이나 평균 {avg_err_hip}도로 너무 {hip_state}졌고, 무릎 각도는 {int(cfg['ref_knee'])}도이나 평균 {avg_err_knee}도로 너무 {knee_state}졌습니다.
        명령: 이 '{cfg['name_en']}' 동작에 특화된 친절하고 구체적인 자세 교정 피드백을 단 2문장 이내로 직관적이게 작성해주세요.
        """

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
        data = {"contents": [{"parts": [{"text": llm_prompt}]}]}
        req = urllib.request.Request(api_url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

        feedback_text = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    feedback_text = result['candidates'][0]['content']['parts'][0]['text']
                break
            except urllib.error.HTTPError as e:
                # 503/429/500/502/504 등 일시적 서버 오류만 재시도
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    print(f"⏳ 서버 일시 오류(HTTP {e.code}). {wait}초 후 재시도... ({attempt}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"❌ API 호출 중 오류 발생: HTTP {e.code} {e.reason}")
                break
            except Exception as e:
                print(f"❌ API 호출 중 오류 발생: {e}")
                break

        if feedback_text:
            print("==================================================")
            print(f"🎙️ [초고속 AI 트레이너 피드백 - {cfg['name_en']}]")
            print(feedback_text.strip())
            print("==================================================")
            error_logs[current_pose] = {"hip": [], "knee": []}

    elif key == ord('q'):
        print("👋 프로그램을 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
