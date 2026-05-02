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

# 자세별 기준 각도 (단위: 도). 골든 스탠다드 데이터로 추후 튜닝 필요.
# landmarks: 코칭 화면에서 표시할 관절 인덱스 (자세별 시각 강조용)
# connections: 그 관절들을 잇는 선 (스틱 형태로 자세 라인 강조)
# MediaPipe Pose 인덱스 — 0:코, 11:좌어깨, 13:좌팔꿈치, 15:좌손목,
#                       23:좌엉덩이, 25:좌무릎, 27:좌발목
POSE_CONFIG = {
    "The_Seal": {
        "ref_hip": 80.0, "ref_knee": 35.0, "tolerance": 15.0,
        "name_kr": "더 씰", "name_en": "The Seal",
        # 작게 말린 C-커브를 보여주기 위해 머리(코)~팔(팔꿈치/손목)까지 포함
        "landmarks":   [0, 11, 13, 15, 23, 25, 27],
        "connections": [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27)],
    },
    "Spine_Stretch": {
        "ref_hip": 80.0, "ref_knee": 175.0, "tolerance": 15.0,
        "name_kr": "스파인 스트레치", "name_en": "Spine Stretch",
        # 앞으로 숙인 정도(코)와 펴진 다리 라인 강조
        "landmarks":   [0, 11, 23, 25, 27],
        "connections": [(11, 23), (23, 25), (25, 27)],
    },
    "Bridging": {
        "ref_hip": 170.0, "ref_knee": 90.0, "tolerance": 15.0,
        "name_kr": "브릿징", "name_en": "Bridging",
        # 누운 자세이므로 머리는 빼고 어깨~발목 직선 라인만
        "landmarks":   [11, 23, 25, 27],
        "connections": [(11, 23), (23, 25), (25, 27)],
    },
}
POSE_ORDER = ["The_Seal", "Spine_Stretch", "Bridging"]

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

# ==========================================
# 3. UI 박스 유틸
# ==========================================

def get_selection_boxes(image_w, image_h):
    """3개의 자세 선택 박스 (가로로 배치)."""
    box_w = int(image_w * 0.22)
    box_h = int(image_h * 0.30)
    gap = int(image_w * 0.03)
    total_w = box_w * 3 + gap * 2
    start_x = (image_w - total_w) // 2
    y = int(image_h * 0.35)
    boxes = {}
    for i, key in enumerate(POSE_ORDER):
        x1 = start_x + i * (box_w + gap)
        boxes[key] = (x1, y, x1 + box_w, y + box_h)
    return boxes

def get_action_boxes(image_w, image_h):
    """코칭 화면 우측에 세로로 배치된 액션 박스들 (FEEDBACK / RESELECT / QUIT)."""
    box_w = int(image_w * 0.22)
    box_h = int(image_h * 0.13)
    margin = 15
    gap = 10
    x1 = image_w - box_w - margin
    y_fb = margin + 40  # 상단 텍스트와 겹치지 않게
    y_re = y_fb + box_h + gap
    y_q  = y_re + box_h + gap
    return {
        "FEEDBACK": (x1, y_fb, x1 + box_w, y_fb + box_h),
        "RESELECT": (x1, y_re, x1 + box_w, y_re + box_h),
        "QUIT":     (x1, y_q,  x1 + box_w, y_q  + box_h),
    }

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def draw_box(image, box, label, progress, active, base_color=(180, 180, 180), highlight_color=(0, 200, 255)):
    x1, y1, x2, y2 = box
    color = highlight_color if active else base_color
    thickness = 3 if active else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # 진행 바 (active 시)
    if active and progress > 0:
        bar_y = y2 - 10
        bar_x_end = int(x1 + (x2 - x1) * progress)
        cv2.rectangle(image, (x1, bar_y), (bar_x_end, y2), (0, 255, 0), -1)
    # 라벨 (박스 중앙)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2 - 10
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# 주먹 클릭 — 양손 모두 사용 가능
# MediaPipe Pose 인덱스: 13/14 = 좌/우 팔꿈치, 15/16 = 좌/우 손목,
#                       17/18 = 좌/우 새끼, 19/20 = 좌/우 검지, 21/22 = 좌/우 엄지
FIST_HANDS = [
    {"name": "left",  "wrist": 15, "elbow": 13, "fingers": [17, 19, 21]},
    {"name": "right", "wrist": 16, "elbow": 14, "fingers": [18, 20, 22]},
]
# 손가락 끝 평균 거리 / 손목-팔꿈치 거리. 이 비율 미만이면 주먹 쥔 상태로 판정.
# 카메라 거리에 영향 받지 않도록 팔뚝 길이로 정규화.
# 펴진 손 ≈ 0.55~0.85, 주먹 ≈ 0.10~0.25.
FIST_RATIO_THRESHOLD = 0.40
# 주먹 → 펴는(release) 순간에 클릭 발생 — 마우스 클릭과 동일한 UX.

fist_state = {"left": False, "right": False}      # 직전 프레임에 주먹이었는가
fist_start_pos = {"left": None, "right": None}    # 주먹 시작 시점의 포인터(손목) 좌표

def get_fist_pointer(landmarks, image_w, image_h, hand_cfg):
    """반환: (포인터(손목 좌표) | None, 주먹 여부 bool)."""
    wrist = landmarks[hand_cfg["wrist"]]
    elbow = landmarks[hand_cfg["elbow"]]
    if getattr(wrist, "visibility", 0.0) < 0.5 or getattr(elbow, "visibility", 0.0) < 0.5:
        return None, False
    wx, wy = int(wrist.x * image_w), int(wrist.y * image_h)
    ex, ey = int(elbow.x * image_w), int(elbow.y * image_h)
    forearm_len = np.hypot(wx - ex, wy - ey)
    if forearm_len < 1:
        return None, False
    finger_dists = []
    for fi in hand_cfg["fingers"]:
        f = landmarks[fi]
        fx, fy = int(f.x * image_w), int(f.y * image_h)
        finger_dists.append(np.hypot(fx - wx, fy - wy))
    avg_finger_dist = float(np.mean(finger_dists))
    ratio = avg_finger_dist / forearm_len
    return (wx, wy), ratio < FIST_RATIO_THRESHOLD

def update_fist_click(landmarks, image_w, image_h):
    """양손 모두 검사. 반환: (시각화용 포인터 리스트, 클릭 발생 좌표 or None)."""
    pointers = []
    click_pos = None
    for cfg in FIST_HANDS:
        name = cfg["name"]
        if landmarks is None:
            fist_state[name] = False
            fist_start_pos[name] = None
            continue
        pointer, is_fist = get_fist_pointer(landmarks, image_w, image_h, cfg)
        if pointer is None:
            fist_state[name] = False
            fist_start_pos[name] = None
            continue
        pointers.append((pointer, is_fist))
        was_fist = fist_state[name]
        if is_fist and not was_fist:
            fist_start_pos[name] = pointer
        elif not is_fist and was_fist:
            if click_pos is None:
                click_pos = fist_start_pos[name]
            fist_start_pos[name] = None
        fist_state[name] = is_fist
    return pointers, click_pos

def reset_fist():
    for cfg in FIST_HANDS:
        fist_state[cfg["name"]] = False
        fist_start_pos[cfg["name"]] = None

# ==========================================
# 4. 상태 변수
# ==========================================
STATE_SELECTION = "selection"
STATE_COACHING = "coaching"
state = STATE_SELECTION

selected_pose = None  # 선택된 자세 키 (예: "The_Seal")

# 코칭 단계용 버퍼
buffer_window = 15
pose_buffer = collections.deque(maxlen=buffer_window)
error_log = {"hip": [], "knee": []}

# ==========================================
# 5. 실시간 웹캠 루프
# ==========================================
cap = cv2.VideoCapture(0)

print("\n" + "*"*50)
print("🎥 실시간 자세 분석 시작")
print("- [선택 화면] 박스 안에서 주먹을 쥐었다 펴면(클릭) 자세가 선택됩니다. 양손 모두 가능.")
print("- [코칭 화면] 자세를 취한 뒤 우측 상단 'FEEDBACK' 박스에 손을 1.5초 올리면 AI 피드백이 생성됩니다.")
print("- 'r' 키: 자세 다시 선택 / 'f' 키: 피드백 즉시 생성 (백업) / 'q' 키: 종료")
print("*"*50)

def request_feedback(pose_key, logs):
    """Gemini REST API 호출 → 피드백 텍스트 반환 (실패 시 None)."""
    cfg = POSE_CONFIG[pose_key]
    error_frames = len(logs["hip"])
    if error_frames < 5:
        print(f"💡 [{cfg['name_en']}] 충분한 오류 데이터가 모이지 않았습니다. ({error_frames}/5)")
        return None

    print(f"\n🧠 [{cfg['name_en']}] 동작을 구글 클라우드 AI가 분석 중입니다... (1~2초 소요)\n")
    avg_err_hip = int(np.mean(logs["hip"]))
    avg_err_knee = int(np.mean(logs["knee"]))
    hip_state = "굽혀" if avg_err_hip < cfg["ref_hip"] else "펴"
    knee_state = "굽혀" if avg_err_knee < cfg["ref_knee"] else "펴"

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
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** (attempt - 1)
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
    return feedback_text


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = detector.detect(mp_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    now = time.time()
    landmarks = results.pose_landmarks[0] if len(results.pose_landmarks) > 0 else None

    # ----------------------------------------
    # 단계 1: 자세 선택 (주먹 클릭)
    # ----------------------------------------
    if state == STATE_SELECTION:
        boxes = get_selection_boxes(w, h)

        # 양손 주먹 검사 — 주먹을 폈을 때(release) click 발생
        pointers, click_pos = update_fist_click(landmarks, w, h)

        # 현재 어느 박스 위에 포인터(손목)가 있는지 (시각 강조용)
        hovering = None
        for (px, py), _ in pointers:
            for key, box in boxes.items():
                if point_in_box(px, py, box):
                    hovering = key
                    break
            if hovering:
                break

        # 박스 그리기 (hover 진행바 없음 — 클릭 즉시 트리거)
        for key, box in boxes.items():
            label = POSE_CONFIG[key]["name_en"]
            is_active = (key == hovering)
            draw_box(image, box, label, 0.0, is_active)

        # 포인터(손목) 표시 — 주먹 쥐면 빨간 원 채움, 펴면 노란 외곽선
        for (px, py), is_fist in pointers:
            if is_fist:
                cv2.circle(image, (px, py), 14, (0, 0, 255), -1)
                cv2.circle(image, (px, py), 20, (0, 0, 255), 2)
            else:
                cv2.circle(image, (px, py), 12, (0, 255, 255), 2)

        cv2.putText(image, "Make a fist inside a box, then open to click",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 클릭이 박스 안에서 발생했으면 선택 처리
        if click_pos:
            for key, box in boxes.items():
                if point_in_box(click_pos[0], click_pos[1], box):
                    selected_pose = key
                    print(f"✅ 자세 선택됨: {POSE_CONFIG[selected_pose]['name_en']}")
                    state = STATE_COACHING
                    pose_buffer.clear()
                    error_log = {"hip": [], "knee": []}
                    reset_fist()
                    break

    # ----------------------------------------
    # 단계 2: 코칭 (오류 누적 + 모션 트리거 피드백)
    # ----------------------------------------
    elif state == STATE_COACHING:
        cfg = POSE_CONFIG[selected_pose]
        cur_hip, cur_knee = 0, 0
        status, color = "[Detecting...]", (200, 200, 200)
        err_count = len(error_log["hip"])

        if landmarks is not None:
            try:
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
                if abs(cur_hip - cfg["ref_hip"]) > cfg["tolerance"] or abs(cur_knee - cfg["ref_knee"]) > cfg["tolerance"]:
                    status, color = "[WARNING]", (0, 0, 255)
                    error_log["hip"].append(cur_hip)
                    error_log["knee"].append(cur_knee)
                err_count = len(error_log["hip"])

                # 자세별 관절/연결선 그리기 — 평가에 의미 있는 점만 강조
                pt_color = (0, 0, 255) if status == "[WARNING]" else (255, 0, 0)
                pose_landmarks = cfg.get("landmarks", [11, 23, 25, 27])
                pose_connections = cfg.get("connections", [])
                pts = {}
                for idx in pose_landmarks:
                    lm = landmarks[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts[idx] = (cx, cy)
                for a, b in pose_connections:
                    if a in pts and b in pts:
                        cv2.line(image, pts[a], pts[b], pt_color, 2)
                for (cx, cy) in pts.values():
                    cv2.circle(image, (cx, cy), 5, pt_color, -1)
            except Exception:
                pass

        # 상단 정보
        cv2.putText(image, f"Pose: {cfg['name_en']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(image, f"Status: {status}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(image, f"Hip: {int(cur_hip)} (Ref: {int(cfg['ref_hip'])})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Knee: {int(cur_knee)} (Ref: {int(cfg['ref_knee'])})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Error Frames: {err_count}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(image, "Make a fist inside FEEDBACK/RESELECT/QUIT, then open  (or [r]/[q])", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 액션 박스 (FEEDBACK / RESELECT / QUIT) — 주먹 클릭
        action_boxes = get_action_boxes(w, h)
        pointers, click_pos = update_fist_click(landmarks, w, h)

        # 현재 어느 박스 위에 포인터가 있는지 (시각 강조용)
        active_target = None
        for (px, py), _ in pointers:
            for key, box in action_boxes.items():
                if point_in_box(px, py, box):
                    active_target = key
                    break
            if active_target:
                break

        for key, box in action_boxes.items():
            is_active = (key == active_target)
            draw_box(image, box, key, 0.0, is_active)

        # 포인터(손목) — 주먹이면 빨강 채움, 펴면 노랑 외곽선
        for (px, py), is_fist in pointers:
            if is_fist:
                cv2.circle(image, (px, py), 14, (0, 0, 255), -1)
                cv2.circle(image, (px, py), 20, (0, 0, 255), 2)
            else:
                cv2.circle(image, (px, py), 12, (0, 255, 255), 2)

        # 클릭이 박스 안에서 발생했으면 해당 액션 실행
        clicked_target = None
        if click_pos:
            for key, box in action_boxes.items():
                if point_in_box(click_pos[0], click_pos[1], box):
                    clicked_target = key
                    break

        if clicked_target == "FEEDBACK":
            request_feedback(selected_pose, error_log)
            error_log = {"hip": [], "knee": []}
            reset_fist()
        elif clicked_target == "RESELECT":
            state = STATE_SELECTION
            selected_pose = None
            pose_buffer.clear()
            error_log = {"hip": [], "knee": []}
            reset_fist()
            print("🔄 자세 선택 화면으로 돌아갑니다.")
        elif clicked_target == "QUIT":
            print("👋 프로그램을 종료합니다.")
            reset_fist()
            break

    cv2.imshow('On-device Pilates Coach (API)', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("👋 프로그램을 종료합니다.")
        break
    elif key == ord('r'):
        # 자세 다시 선택
        state = STATE_SELECTION
        selected_pose = None
        pose_buffer.clear()
        error_log = {"hip": [], "knee": []}
        reset_fist()
        print("🔄 자세 선택 화면으로 돌아갑니다.")
    elif key == ord('f') and state == STATE_COACHING:
        # 백업: 키보드 트리거도 유지
        request_feedback(selected_pose, error_log)
        error_log = {"hip": [], "knee": []}

cap.release()
cv2.destroyAllWindows()
