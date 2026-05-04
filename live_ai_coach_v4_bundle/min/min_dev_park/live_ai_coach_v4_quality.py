from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
LIFTER_DIR = PROJECT_ROOT / "pilates_temporal_lifter"
sys.path.insert(0, str(LIFTER_DIR))

from dataset import JOINT_ORDER  # noqa: E402
from runtime_lifting import OnlineTemporalLifter  # noqa: E402
from runtime_quality import format_summary_for_prompt, load_scorer, score_sequence_3d, summarize_score_rows  # noqa: E402


ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PASTE_YOUR_KEY_HERE":
    raise RuntimeError(f"GOOGLE_API_KEY is not set in {ENV_PATH}")


POSE_CONFIG = {
    "The_Seal": {
        "name_en": "The Seal",
        "name_kr": "더 씰",
        "scorer": LIFTER_DIR / "runs" / "the_seal_mahalanobis_hip_knee_all_v2" / "model.json",
        "future_feature_set": "hip_knee",
        "landmarks": [0, 11, 13, 15, 23, 25, 27],
        "connections": [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27)],
        "fallback_ref": {"hip": 80.0, "knee": 35.0, "trunk": None, "tolerance": 15.0},
    },
    "Spine_Stretch": {
        "name_en": "Spine Stretch",
        "name_kr": "스파인 스트레치",
        "scorer": None,
        "future_feature_set": "spine_stretch",
        "landmarks": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        "connections": [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)],
        "fallback_ref": {"hip": 80.0, "knee": 175.0, "trunk": None, "tolerance": 15.0},
    },
    "Bridging": {
        "name_en": "Bridging",
        "name_kr": "브릿징",
        "scorer": LIFTER_DIR / "runs" / "bridging_mahalanobis_v1" / "model.json",
        "future_feature_set": "bridging",
        "landmarks": [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        "connections": [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)],
        "fallback_ref": {"hip": 170.0, "knee": 90.0, "trunk": None, "tolerance": 15.0},
    },
}
POSE_ORDER = ["The_Seal", "Spine_Stretch", "Bridging"]

LIFTER_CHECKPOINT = LIFTER_DIR / "runs" / "the_seal_progress3_angle_causal_v1" / "best.pt"
if not LIFTER_CHECKPOINT.exists():
    LIFTER_CHECKPOINT = LIFTER_DIR / "runs" / "the_seal_progress3_lift_only_v1" / "best.pt"

STATE_SELECTION = "selection"
STATE_WAIT_START = "wait_start"
STATE_ACTIVE = "active"
STATE_REPORT = "report"

HOLD_SECONDS = 1.4
HOLD_STILL_PIXELS = 32.0
CONTROL_COOLDOWN_SECONDS = 0.45
MIN_REPORT_FRAMES = 20

JOINT_IDX = {name: idx for idx, name in enumerate(JOINT_ORDER)}


class HoldTracker:
    def __init__(self, hold_seconds: float, still_pixels: float) -> None:
        self.hold_seconds = hold_seconds
        self.still_pixels = still_pixels
        self.target = None
        self.anchor = None
        self.started_at = None

    def reset(self) -> None:
        self.target = None
        self.anchor = None
        self.started_at = None

    def update(self, target: str | None, pointer: tuple[int, int] | None, now: float) -> tuple[float, bool]:
        if target is None or pointer is None:
            self.reset()
            return 0.0, False
        pointer_arr = np.asarray(pointer, dtype=np.float32)
        if self.target != target or self.anchor is None:
            self.target = target
            self.anchor = pointer_arr
            self.started_at = now
            return 0.0, False
        if float(np.linalg.norm(pointer_arr - self.anchor)) > self.still_pixels:
            self.anchor = pointer_arr
            self.started_at = now
            return 0.0, False
        progress = min(1.0, (now - float(self.started_at)) / self.hold_seconds)
        if progress >= 1.0:
            self.reset()
            return 1.0, True
        return progress, False


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def lifted_angles(frame3d: np.ndarray | None) -> dict[str, float]:
    if frame3d is None:
        return {"hip": 0.0, "knee": 0.0, "trunk": 0.0}
    lhip = angle_deg(frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]])
    rhip = angle_deg(frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]])
    lknee = angle_deg(frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]], frame3d[JOINT_IDX["LAnkle"]])
    rknee = angle_deg(frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]], frame3d[JOINT_IDX["RAnkle"]])
    trunk = angle_deg(frame3d[JOINT_IDX["Hip"]], frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["Head"]])
    return {"hip": (lhip + rhip) / 2.0, "knee": (lknee + rknee) / 2.0, "trunk": trunk}


def get_selection_boxes(image_w: int, image_h: int) -> dict[str, tuple[int, int, int, int]]:
    box_w = int(image_w * 0.22)
    box_h = int(image_h * 0.30)
    gap = int(image_w * 0.03)
    total_w = box_w * 3 + gap * 2
    start_x = (image_w - total_w) // 2
    y = int(image_h * 0.35)
    return {
        key: (start_x + i * (box_w + gap), y, start_x + i * (box_w + gap) + box_w, y + box_h)
        for i, key in enumerate(POSE_ORDER)
    }


def get_control_boxes(image_w: int, image_h: int, mode: str) -> dict[str, tuple[int, int, int, int]]:
    box_w = int(image_w * 0.22)
    box_h = int(image_h * 0.13)
    margin = 15
    gap = 10
    x1 = image_w - box_w - margin
    y = margin + 40
    labels = ["START", "RESELECT", "QUIT"] if mode == STATE_WAIT_START else ["FINISH", "RESELECT", "QUIT"]
    return {
        label: (x1, y + i * (box_h + gap), x1 + box_w, y + i * (box_h + gap) + box_h)
        for i, label in enumerate(labels)
    }


def point_in_box(px: int, py: int, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_box(
    image,
    box: tuple[int, int, int, int],
    label: str,
    progress: float = 0.0,
    active: bool = False,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 220, 255) if active else (170, 170, 170)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if active else 2)
    if active and progress > 0:
        fill_w = int((x2 - x1) * progress)
        cv2.rectangle(image, (x1, y2 - 12), (x1 + fill_w, y2), (0, 220, 0), -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(image, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def get_hand_pointers(landmarks, image_w: int, image_h: int) -> list[tuple[int, int]]:
    if landmarks is None:
        return []
    out = []
    for idx in (15, 16):
        lm = landmarks[idx]
        if getattr(lm, "visibility", 0.0) >= 0.5:
            out.append((int(lm.x * image_w), int(lm.y * image_h)))
    return out


FIST_HANDS = [
    {"name": "left", "wrist": 15, "elbow": 13, "fingers": [17, 19, 21]},
    {"name": "right", "wrist": 16, "elbow": 14, "fingers": [18, 20, 22]},
]
FIST_RATIO_THRESHOLD = 0.40
fist_state = {"left": False, "right": False}
fist_start_pos = {"left": None, "right": None}


def get_fist_pointer(landmarks, image_w: int, image_h: int, cfg: dict) -> tuple[tuple[int, int] | None, bool]:
    wrist = landmarks[cfg["wrist"]]
    elbow = landmarks[cfg["elbow"]]
    if getattr(wrist, "visibility", 0.0) < 0.5 or getattr(elbow, "visibility", 0.0) < 0.5:
        return None, False
    wx, wy = int(wrist.x * image_w), int(wrist.y * image_h)
    ex, ey = int(elbow.x * image_w), int(elbow.y * image_h)
    forearm_len = np.hypot(wx - ex, wy - ey)
    if forearm_len < 1:
        return None, False
    finger_dists = []
    for finger_idx in cfg["fingers"]:
        finger = landmarks[finger_idx]
        finger_dists.append(np.hypot(int(finger.x * image_w) - wx, int(finger.y * image_h) - wy))
    return (wx, wy), float(np.mean(finger_dists)) / forearm_len < FIST_RATIO_THRESHOLD


def update_fist_click(landmarks, image_w: int, image_h: int):
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
        if is_fist and not fist_state[name]:
            fist_start_pos[name] = pointer
        elif not is_fist and fist_state[name]:
            if click_pos is None:
                click_pos = fist_start_pos[name]
            fist_start_pos[name] = None
        fist_state[name] = is_fist
    return pointers, click_pos


def reset_fist() -> None:
    for cfg in FIST_HANDS:
        fist_state[cfg["name"]] = False
        fist_start_pos[cfg["name"]] = None


def active_hold_target(pointers: list[tuple[int, int]], boxes: dict[str, tuple[int, int, int, int]]):
    for pointer in pointers:
        for label, box in boxes.items():
            if point_in_box(pointer[0], pointer[1], box):
                return label, pointer
    return None, None


def request_feedback(pose_key: str, report_summary: dict, scorer_available: bool) -> str | None:
    cfg = POSE_CONFIG[pose_key]
    report_text = format_summary_for_prompt(cfg["name_en"], report_summary)
    mode_note = (
        "The report is based on phase-wise Mahalanobis scoring over lifted 3D hip/knee kinematic features."
        if scorer_available
        else "No trained Mahalanobis scorer is available for this pose yet; use the fallback angle notes cautiously."
    )
    prompt = f"""
You are a precise but encouraging Pilates coach. Answer in Korean.
Pose: {cfg['name_en']} ({cfg['name_kr']})
Scoring mode: {mode_note}
Analysis:
{report_text}

Write a concise feedback report in 3 sentences:
1. Overall quality.
2. The most important correction based on the top issue feature or phase.
3. A simple cue the user can try on the next attempt.
Do not mention implementation details unless needed.
"""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(api_url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print("\n" + "=" * 60)
            print(f"[AI Coach Feedback - {cfg['name_en']}]")
            print(text)
            print("=" * 60)
            return text
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** (attempt - 1))
                continue
            print(f"Gemini API error: HTTP {exc.code} {exc.reason}")
            return None
        except Exception as exc:
            print(f"Gemini API error: {exc}")
            return None
    return None


def fallback_summary(error_events: list[dict], frame_count: int) -> dict:
    return {
        "frames": int(frame_count),
        "status_counts": {"WARN": len(error_events)} if error_events else {"PASS": frame_count},
        "phase_counts": {},
        "top_issue_counts": {
            key: sum(1 for event in error_events if event["feature"] == key)
            for key in sorted({event["feature"] for event in error_events})
        },
        "mean_mahalanobis_d2": 0.0,
        "max_mahalanobis_d2": 0.0,
        "warn_or_fail_ratio": float(len(error_events) / max(frame_count, 1)),
        "worst_frames": error_events[-8:],
    }


def draw_pose_lines(image, landmarks, cfg: dict, color: tuple[int, int, int]) -> None:
    if landmarks is None:
        return
    h, w = image.shape[:2]
    pts = {}
    for idx in cfg.get("landmarks", []):
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < 0.2:
            continue
        pts[idx] = (int(lm.x * w), int(lm.y * h))
    for a, b in cfg.get("connections", []):
        if a in pts and b in pts:
            cv2.line(image, pts[a], pts[b], color, 2)
    for point in pts.values():
        cv2.circle(image, point, 5, color, -1)


def main() -> int:
    print("=" * 60)
    print("OnPose live quality coach v4")
    print("3D lifting + phase-wise kinematic scoring + LLM report")
    print("=" * 60)

    model_path = PROJECT_ROOT / "pose_landmarker_heavy.task"
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)

    with open(model_path, "rb") as f:
        model_buffer = f.read()
    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_buffer),
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )

    lifter = None
    if LIFTER_CHECKPOINT.exists():
        lifter = OnlineTemporalLifter(LIFTER_CHECKPOINT, window_size=81)
        print(f"Loaded lifter: {LIFTER_CHECKPOINT}")
    else:
        print("No lifter checkpoint found. Falling back to non-scored angle preview.")

    scorers = {key: load_scorer(cfg.get("scorer")) for key, cfg in POSE_CONFIG.items()}
    for key, scorer in scorers.items():
        print(f"Scorer {POSE_CONFIG[key]['name_en']}: {'loaded' if scorer else 'not available'}")

    cap = cv2.VideoCapture(0)
    state = STATE_SELECTION
    selected_pose = None
    hold = HoldTracker(HOLD_SECONDS, HOLD_STILL_PIXELS)
    session_3d: list[np.ndarray] = []
    fallback_errors: list[dict] = []
    last_report = None
    last_feedback = None
    last_control_time = 0.0
    latest_angles = {"hip": 0.0, "knee": 0.0, "trunk": 0.0}
    latest_status = "Idle"

    def reset_session() -> None:
        nonlocal session_3d, fallback_errors, last_report, last_feedback, latest_status
        session_3d = []
        fallback_errors = []
        last_report = None
        last_feedback = None
        latest_status = "Recording"
        if lifter is not None:
            lifter.reset()

    def finish_session() -> None:
        nonlocal state, last_report, last_feedback, latest_status
        cfg = POSE_CONFIG[selected_pose]
        scorer = scorers.get(selected_pose)
        if scorer is not None and len(session_3d) >= MIN_REPORT_FRAMES:
            rows = score_sequence_3d(np.asarray(session_3d, dtype=np.float32), scorer, decision_window=5)
            last_report = summarize_score_rows(rows)
            latest_status = f"Report ready: {last_report.get('status_counts', {})}"
            last_feedback = request_feedback(selected_pose, last_report, scorer_available=True)
        else:
            last_report = fallback_summary(fallback_errors, len(session_3d))
            latest_status = f"Fallback report: {last_report.get('status_counts', {})}"
            last_feedback = request_feedback(selected_pose, last_report, scorer_available=False)
        state = STATE_REPORT

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        now = time.time()
        landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        if state == STATE_SELECTION:
            boxes = get_selection_boxes(w, h)
            pointers, click_pos = update_fist_click(landmarks, w, h)
            hover = None
            for pointer, _ in pointers:
                for key, box in boxes.items():
                    if point_in_box(pointer[0], pointer[1], box):
                        hover = key
                        break
            for key, box in boxes.items():
                draw_box(image, box, POSE_CONFIG[key]["name_en"], active=(key == hover))
            for pointer, is_fist in pointers:
                cv2.circle(image, pointer, 14, (0, 0, 255) if is_fist else (0, 255, 255), -1 if is_fist else 2)
            cv2.putText(image, "Select pose: make a fist in a box, then open", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if click_pos:
                for key, box in boxes.items():
                    if point_in_box(click_pos[0], click_pos[1], box):
                        selected_pose = key
                        state = STATE_WAIT_START
                        reset_session()
                        hold.reset()
                        reset_fist()
                        print(f"Selected pose: {POSE_CONFIG[key]['name_en']}")
                        break

        elif state in (STATE_WAIT_START, STATE_ACTIVE, STATE_REPORT):
            cfg = POSE_CONFIG[selected_pose]
            draw_pose_lines(image, landmarks, cfg, (255, 0, 0))
            boxes = get_control_boxes(w, h, STATE_WAIT_START if state == STATE_WAIT_START else STATE_ACTIVE)
            pointers = get_hand_pointers(landmarks, w, h)
            target, pointer = active_hold_target(pointers, boxes)
            progress, completed = hold.update(target, pointer, now)
            control_active = target is not None

            for label, box in boxes.items():
                draw_box(image, box, label, progress if label == target else 0.0, active=(label == target))
            for pointer_item in pointers:
                cv2.circle(image, pointer_item, 12, (0, 255, 255), 2)

            if state == STATE_WAIT_START:
                latest_status = "Hold START still to begin"
                if completed and target == "START":
                    reset_session()
                    state = STATE_ACTIVE
                    last_control_time = now
                    print("Session started")
                elif completed and target == "RESELECT":
                    state = STATE_SELECTION
                    selected_pose = None
                    hold.reset()
                elif completed and target == "QUIT":
                    break

            elif state == STATE_ACTIVE:
                if completed and target == "FINISH":
                    last_control_time = now
                    print("Session finished. Scoring clean exercise frames only.")
                    finish_session()
                elif completed and target == "RESELECT":
                    state = STATE_SELECTION
                    selected_pose = None
                    hold.reset()
                    reset_session()
                elif completed and target == "QUIT":
                    break
                else:
                    should_log = (
                        landmarks is not None
                        and not control_active
                        and now - last_control_time > CONTROL_COOLDOWN_SECONDS
                    )
                    if should_log and lifter is not None:
                        latest_3d, _ = lifter.append_landmarks(landmarks)
                        if latest_3d is not None:
                            session_3d.append(latest_3d)
                            latest_angles = lifted_angles(latest_3d)
                            latest_status = "Recording clean movement"
                            if scorers.get(selected_pose) is None:
                                ref = cfg["fallback_ref"]
                                for feature in ("hip", "knee", "trunk"):
                                    if ref.get(feature) is None:
                                        continue
                                    if abs(latest_angles[feature] - ref[feature]) > ref["tolerance"]:
                                        fallback_errors.append(
                                            {
                                                "frame": len(session_3d),
                                                "feature": feature,
                                                "value": round(latest_angles[feature], 2),
                                                "ref": ref[feature],
                                            }
                                        )
                    elif control_active:
                        latest_status = "Control hold active: logging paused"

            elif state == STATE_REPORT:
                if completed and target == "START":
                    reset_session()
                    state = STATE_ACTIVE
                    last_control_time = now
                    print("New session started")
                elif completed and target == "RESELECT":
                    state = STATE_SELECTION
                    selected_pose = None
                    hold.reset()
                elif completed and target == "QUIT":
                    break

            cv2.putText(image, f"Pose: {cfg['name_en']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(image, f"State: {state}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Status: {latest_status}", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            cv2.putText(image, f"Clean frames: {len(session_3d)}", (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(image, f"Hip {latest_angles['hip']:.1f}  Knee {latest_angles['knee']:.1f}  Trunk {latest_angles['trunk']:.1f}", (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if last_report:
                cv2.putText(image, f"Report: {last_report.get('status_counts', {})}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

        cv2.imshow("OnPose Live Quality Coach v4", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state = STATE_SELECTION
            selected_pose = None
            hold.reset()
            reset_session()
        if key == ord("f") and selected_pose is not None:
            if state == STATE_ACTIVE:
                finish_session()
            elif last_report:
                last_feedback = request_feedback(selected_pose, last_report, scorers.get(selected_pose) is not None)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
