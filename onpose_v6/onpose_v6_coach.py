"""
OnPose Live Coach v6 — Light 테마 (NCCOSS 디자인 참고).

CLI:
  --offline       LLM 없이 친근체 템플릿
  --lite          모바일 시뮬레이션
  --no-lifter / --no-bone-lock / --no-smooth / --no-distribution
  --variant {lite,full,heavy}
  --camera N
  --video PATH
  --record OUT
  --demo
  --voice         TTS 음성 (별로 추천 안 함 — 한국어 Heami만 있으면 부자연스러움)
  --sound         사운드 효과만 (시작/종료/점수 비프음 — 권장)
  --calibrated
  --no-expert / --no-replay / --no-splash
  --show-inpainted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from core.angle_scorer import (RUBRICS, apply_calibrated_rubrics, get_rubric,
                                score_sequence, score_single_frame)
from core.distribution_scorer import (load_distribution_rubrics, score_frame_distribution,
                                       score_sequence_distribution)
from core.feedback_engine import generate_feedback
from core.latency import LatencyProfiler
from core.light_ui import (L_AMBER, L_BG, L_BLUE, L_CARD, L_DIVIDER, L_GREEN, L_GREEN_BG,
                            L_GREEN_DARK, L_HEADER_BG, L_RED, L_SUB, L_TEXT,
                            draw_action_button, draw_camera_frame, draw_feedback_row,
                            draw_header, draw_metric_row, draw_mode_strip, draw_pose_mini,
                            draw_score_pill, draw_workout_info_bar, kr, kr_w, rounded_rect,
                            soft_shadow_card)
from core.minimal_ui import _draw_pose_icon_minimal
from core.occlusion_robust import FrameRecorder
from core.pose_classifier import classify_pose
from core.pose_guide import POSE_HINTS, POSE_SAFETY, POSE_TIPS
from core.pose_pipeline import (JOINT_IDX, PoseAnglePipeline, build_detector,
                                build_lifter, find_pose_landmarker)
from core.voice import VoiceCoach

PROJECT_ROOT = THIS_DIR.parent
REPORTS_DIR = THIS_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = THIS_DIR / "assets"
HIST_FILE = REPORTS_DIR / "session_history.json"

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

POSE_ORDER = ["The_Seal", "Spine_Stretch", "Bridging"]
POSE_LABELS = {"The_Seal": "하체 운동: 더 씰",
               "Spine_Stretch": "전신 운동: 스파인 스트레치",
               "Bridging": "하체 운동: 브릿징"}

S_SPLASH = "splash"; S_SELECT = "select"; S_INTRO = "intro"
S_CAPTURE = "capture"; S_PROCESSING = "processing"; S_RESULT = "result"

HOLD_SECONDS = 1.4
INTRO_SECONDS = 3.0
CAPTURE_SECONDS = 5.0
SPLASH_SECONDS = 1.5
CANVAS_W, CANVAS_H = 1280, 800


class HoldButton:
    def __init__(self, hold_seconds: float = HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.target: Optional[str] = None
        self.started: Optional[float] = None

    def reset(self):
        self.target = None; self.started = None

    def update(self, target: Optional[str], now: float) -> Tuple[float, bool]:
        if target is None:
            self.reset(); return 0.0, False
        if self.target != target:
            self.target = target; self.started = now
            return 0.0, False
        progress = min(1.0, (now - float(self.started)) / self.hold_seconds)
        if progress >= 1.0:
            self.reset(); return 1.0, True
        return progress, False


def get_hand_pointers(landmarks, w: int, h: int) -> List[Tuple[int, int]]:
    if landmarks is None:
        return []
    out = []
    for idx in (15, 16):
        lm = landmarks[idx]
        if getattr(lm, "visibility", 0.0) >= 0.45:
            out.append((int(lm.x * w), int(lm.y * h)))
    return out


def point_in(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


# ---------------------------------------------------------------------------
# 전문가 영상 / 사용자 리플레이
# ---------------------------------------------------------------------------
class LoopVideo:
    def __init__(self, path: Path, size: Tuple[int, int]):
        self.cap = cv2.VideoCapture(str(path))
        self.size = size
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")

    def next_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return cv2.resize(frame, self.size)

    def release(self):
        self.cap.release()


def load_expert_video(pose_key: str, size: Tuple[int, int]) -> Optional[LoopVideo]:
    for ext in (".mp4", ".webm", ".mov", ".avi"):
        p = ASSETS_DIR / f"{pose_key}{ext}"
        if p.exists():
            try:
                v = LoopVideo(p, size)
                test = v.next_frame()
                if test is None:
                    v.release()
                    print(f"[expert] {p.name} 코덱 미지원")
                    return None
                v.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[expert] loaded {p.name}")
                return v
            except Exception as e:
                print(f"[expert] {p.name} 열기 실패: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Light 테마 헤더/푸터 공통
# ---------------------------------------------------------------------------
def draw_common_chrome(canvas, runtime_mode: str, state_kr: str,
                      pose_label: Optional[str], elapsed_seconds: float) -> None:
    draw_header(canvas, "실시간 자세 코칭", height=64)
    # 우상단 상태 칩 (state_kr)
    state_color_map = {
        S_SELECT: L_BLUE, S_INTRO: L_AMBER, S_CAPTURE: L_RED,
        S_PROCESSING: L_BLUE, S_RESULT: L_GREEN_DARK, S_SPLASH: L_SUB,
    }
    # 운동 정보 박스 (자세명 + 경과시간) — RESULT/PROCESSING 외엔 표시
    if pose_label:
        mm = int(elapsed_seconds // 60); ss = int(elapsed_seconds % 60)
        draw_workout_info_bar(canvas, pose_label, f"{mm:02d}:{ss:02d}",
                              y=80, height=52)


def fill_bg(canvas):
    canvas[:] = L_BG


# ---------------------------------------------------------------------------
# State별 렌더링
# ---------------------------------------------------------------------------
def render_splash(canvas) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = L_BG
    # 가운데 큰 로고/타이틀
    cx, cy = w // 2, h // 2 - 40
    # 큰 동그라미 + 체크 아이콘
    soft_shadow_card(canvas, (cx - 70, cy - 70), (cx + 70, cy + 70), radius=70, shadow_offset=8, shadow_alpha=0.10)
    cv2.circle(canvas, (cx, cy), 60, L_GREEN_BG, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 60, L_GREEN_DARK, 3, cv2.LINE_AA)
    cv2.line(canvas, (cx - 22, cy + 4), (cx - 6, cy + 20), L_GREEN_DARK, 5, cv2.LINE_AA)
    cv2.line(canvas, (cx - 6, cy + 20), (cx + 26, cy - 14), L_GREEN_DARK, 5, cv2.LINE_AA)
    title = "OnPose Coach v6"
    canvas[:] = kr(canvas, title, (cx - kr_w(title, 38) // 2, cy + 90), size=38, color=L_TEXT)
    sub = "온디바이스 자세 코칭"
    canvas[:] = kr(canvas, sub, (cx - kr_w(sub, 18) // 2, cy + 140), size=18, color=L_SUB)


def render_select(canvas, landmarks, angles, hover_key, progress, history_for_chart) -> None:
    """자세 선택 — 흰 카드 3장 가로 배치"""
    h, w = canvas.shape[:2]
    cy_top = 140
    title = "운동 자세를 선택해 주세요"
    canvas[:] = kr(canvas, title, ((w - kr_w(title, 26)) // 2, 100), size=26, color=L_TEXT)
    sub = "원하는 자세 카드에 손목을 1.4초간 올려 주세요"
    canvas[:] = kr(canvas, sub, ((w - kr_w(sub, 16)) // 2, 140), size=16, color=L_SUB)
    # 카드 3개
    card_w, card_h, gap = 280, 360, 36
    sx = (w - card_w * 3 - gap * 2) // 2
    sy = 200
    boxes = {}
    for i, key in enumerate(POSE_ORDER):
        x = sx + i * (card_w + gap)
        boxes[key] = (x, sy, x + card_w, sy + card_h)
        active = (key == hover_key)
        # 카드
        rad = 20
        soft_shadow_card(canvas, (x, sy), (x + card_w, sy + card_h), radius=rad,
                         shadow_offset=6, shadow_alpha=0.08)
        if active:
            rounded_rect(canvas, (x, sy), (x + card_w, sy + card_h), L_GREEN_DARK, 3, rad)
        # 일러스트 (상단)
        rub = get_rubric(key)
        _draw_pose_icon_minimal(canvas, x + 28, sy + 28, x + card_w - 28, sy + card_h - 130,
                                key, color=(L_GREEN_DARK if active else (170, 180, 195)))
        # 라벨 박스 (하단)
        lab_y1 = sy + card_h - 110; lab_y2 = sy + card_h - 22
        bg = L_GREEN_BG if active else (245, 245, 247)
        rounded_rect(canvas, (x + 18, lab_y1), (x + card_w - 18, lab_y2), bg, -1, 14)
        canvas[:] = kr(canvas, rub.pose_name_kr,
                       (x + (card_w - kr_w(rub.pose_name_kr, 24)) // 2, lab_y1 + 12),
                       size=24, color=L_GREEN_DARK if active else L_TEXT)
        tip = POSE_HINTS.get(key, "")
        if tip:
            canvas[:] = kr(canvas, tip,
                           (x + (card_w - kr_w(tip, 13)) // 2, lab_y1 + 52),
                           size=13, color=L_SUB)
        # 진행바
        if active and progress > 0:
            fw = int((card_w - 36) * progress)
            rounded_rect(canvas, (x + 18, lab_y2 - 6), (x + 18 + fw, lab_y2 - 2),
                         L_GREEN_DARK, -1, 2)

    # 자동 인식 배지
    if landmarks is not None and any(angles.get(k, 0) > 0 for k in angles):
        auto_key, conf, _ = classify_pose(angles)
        if auto_key and conf > 0.4:
            rub = get_rubric(auto_key)
            badge = f"AI 자동 인식: {rub.pose_name_kr}  ({conf*100:.0f}%)"
            bw_total = kr_w(badge, 14) + 80
            bx, by = (w - bw_total) // 2, sy + card_h + 30
            rounded_rect(canvas, (bx, by), (bx + bw_total, by + 36), L_GREEN_BG, -1, 18)
            cv2.circle(canvas, (bx + 18, by + 18), 5, L_GREEN_DARK, -1)
            canvas[:] = kr(canvas, badge, (bx + 32, by + 9), size=14, color=L_GREEN_DARK)
            canvas[:] = kr(canvas, "키보드 = 키를 누르면 바로 시작",
                          ((w - kr_w("키보드 = 키를 누르면 바로 시작", 12)) // 2, by + 44),
                          size=12, color=L_SUB)
    # 하단 모드 스트립
    draw_mode_strip(canvas, h - 60, height=44)
    return boxes


def render_intro(canvas, rubric, remain: float, alignment: Dict) -> None:
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2 - 20
    # 큰 카운트다운 원
    cv2.circle(canvas, (cx, cy), 110, L_CARD, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 110, L_DIVIDER, 2, cv2.LINE_AA)
    # 진행 호
    angle = int(360 * (1 - remain / INTRO_SECONDS))
    if angle > 0:
        cv2.ellipse(canvas, (cx, cy), (110, 110), -90, 0, angle, L_GREEN_DARK, 8, cv2.LINE_AA)
    # 가운데 숫자
    num = f"{remain:.0f}"
    canvas[:] = kr(canvas, num, (cx - kr_w(num, 64) // 2, cy - 42),
                  size=64, color=L_TEXT)
    canvas[:] = kr(canvas, "초 후 시작", (cx - kr_w("초 후 시작", 16) // 2, cy + 36),
                  size=16, color=L_SUB)
    # 자세명
    pose_kr = rubric.pose_name_kr
    canvas[:] = kr(canvas, pose_kr, (cx - kr_w(pose_kr, 32) // 2, cy + 140),
                  size=32, color=L_TEXT)
    # 안내
    canvas[:] = kr(canvas, "자세를 취해 주세요", (cx - kr_w("자세를 취해 주세요", 18) // 2, cy + 190),
                  size=18, color=L_GREEN_DARK)
    # 정렬 상태 (좌하단 카드)
    a_color = L_GREEN_DARK if alignment["score"] >= 0.6 else L_AMBER
    cx1, cy1 = 16, h - 100
    soft_shadow_card(canvas, (cx1, cy1), (cx1 + 380, cy1 + 64), radius=14, shadow_offset=3)
    canvas[:] = kr(canvas, "자세 정렬", (cx1 + 16, cy1 + 8), size=13, color=L_SUB)
    canvas[:] = kr(canvas, alignment["msg"], (cx1 + 16, cy1 + 28), size=16, color=a_color)
    # 정렬 게이지
    cv2.rectangle(canvas, (cx1 + 16, cy1 + 52), (cx1 + 360, cy1 + 58), (235, 235, 240), -1)
    fill = int(344 * alignment["score"])
    cv2.rectangle(canvas, (cx1 + 16, cy1 + 52), (cx1 + 16 + fill, cy1 + 58), a_color, -1)


def render_capture(canvas, rubric, remain: float, angles: Dict, pose_key: str,
                  cam_bgr: np.ndarray, expert_frame: Optional[np.ndarray],
                  captured: int, score_summary_running: Dict,
                  landmarks=None) -> None:
    """레이아웃: 좌 (내 자세, 큰) | 우상 (전문가 영상) | 우하 (점수+각도+카운트다운)"""
    h, w = canvas.shape[:2]
    pose_label = POSE_LABELS.get(pose_key, pose_key)
    elapsed = CAPTURE_SECONDS - remain
    draw_header(canvas, "실시간 자세 코칭", height=64)
    draw_workout_info_bar(canvas, pose_label, f"00:{int(elapsed):02d} / 00:{int(CAPTURE_SECONDS):02d}",
                          y=80, height=52)

    main_y = 148
    main_h = 380
    # 좌측: 내 카메라 (48%)
    cam_w_px = int(w * 0.48)
    # 우측 윗부분: 전문가 영상 카드 (28%)
    expert_w = int(w * 0.28)
    expert_x = 16 + cam_w_px + 14
    expert_h = 220
    # 우측 아래: 정확도 + 각도 패널 (28%)
    info_x = expert_x
    info_y = main_y + expert_h + 12
    info_w = expert_w
    # 우측 끝: 우측 끝의 다이얼 컬럼 (남은 폭 24%)
    dial_x = expert_x + expert_w + 12
    dial_w = w - 16 - dial_x

    # 내 카메라 + 골격 오버레이
    highlight = {"The_Seal": [23, 24, 25, 26],
                 "Spine_Stretch": [23, 24, 25, 26, 27, 28],
                 "Bridging": [23, 24, 25, 26]}.get(pose_key, [])
    draw_camera_frame(canvas, 16, main_y, cam_w_px, main_h, cam_bgr=cam_bgr,
                      label="📹 LIVE  내 자세",
                      landmarks=landmarks, pose_key=pose_key, highlight_indices=highlight)

    # 전문가 영상 카드 (우측 상단)
    soft_shadow_card(canvas, (expert_x, main_y), (expert_x + expert_w, main_y + expert_h),
                     radius=14, shadow_offset=5, shadow_alpha=0.10)
    canvas[:] = kr(canvas, "🎯 전문가 자세", (expert_x + 14, main_y + 10), size=15, color=L_GREEN_DARK)
    inner_x = expert_x + 8; inner_y = main_y + 34
    inner_w = expert_w - 16; inner_h = expert_h - 44
    if expert_frame is not None:
        ef = cv2.resize(expert_frame, (inner_w, inner_h))
        canvas[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w] = ef
    else:
        cv2.rectangle(canvas, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h),
                      (240, 242, 246), -1)
        canvas[:] = kr(canvas, "전문가 영상 준비 중", (inner_x + inner_w // 2 - 70, inner_y + inner_h // 2 - 12),
                       size=14, color=L_SUB)

    # 우측 아래: 정확도 점수
    per = score_summary_running
    score = per.get("score", 0)
    draw_score_pill(canvas, info_x, info_y, info_w, 130, accuracy=int(score),
                    label="자세 정확도",
                    remark="훌륭해요!" if score >= 85 else "좋아요!" if score >= 70 else "조금만 더!")

    # 우측 끝: 각도 다이얼 + 카운트다운
    dy = main_y
    # 카운트다운 카드 (상단)
    soft_shadow_card(canvas, (dial_x, dy), (dial_x + dial_w, dy + 90), radius=14, shadow_offset=3)
    canvas[:] = kr(canvas, "남은 시간", (dial_x + 14, dy + 10), size=13, color=L_SUB)
    big = f"{remain:.1f}s"
    canvas[:] = kr(canvas, big, (dial_x + 14, dy + 28), size=36, color=L_AMBER)
    canvas[:] = kr(canvas, f"수집 {captured} 프레임", (dial_x + 14, dy + 70), size=12, color=L_SUB)
    dy += 100
    # 각도 다이얼들
    for k, spec in rubric.angles.items():
        d = per.get("details", {}).get(k, {})
        cur = d.get("value", angles.get(k, 0))
        ok = d.get("ok", False)
        col = L_GREEN_DARK if ok else L_AMBER
        draw_metric_row(canvas, dial_x, dy, dial_w, 70,
                        spec.name, f"{cur:.0f}°",
                        sub_label=f"목표 {spec.target_deg:.0f}°")
        dy += 78

    # 하단 — 실시간 피드백 카드
    fb_y = main_y + main_h + 16
    fb_w = w - 32
    fb_h = 120
    soft_shadow_card(canvas, (16, fb_y), (16 + fb_w, fb_y + fb_h), radius=14, shadow_offset=4)
    canvas[:] = kr(canvas, "실시간 피드백", (32, fb_y + 14), size=18, color=L_TEXT)
    yy = fb_y + 48
    issues = []
    for k, spec in rubric.angles.items():
        d = per.get("details", {}).get(k, {})
        if not d.get("ok", True):
            v = d.get("value", 0); t = spec.target_deg; diff = d.get("diff", abs(v - t))
            direction = "더 굽혀 주세요" if v < t else "더 펴 주세요"
            issues.append((k, f"{spec.name}이(가) 목표보다 {diff:.0f}° 차이나요. {direction}"))
    if not issues:
        yy = draw_feedback_row(canvas, 32, yy, fb_w - 32, "check",
                                "좋아요! 핵심 각도들이 모두 정답 범위 안에 있어요.")
        yy = draw_feedback_row(canvas, 32, yy, fb_w - 32, "info", POSE_HINTS.get(pose_key, ""))
    else:
        for k, msg in issues[:2]:
            yy = draw_feedback_row(canvas, 32, yy, fb_w - 32, "warn", msg)
    # 모드 스트립
    draw_mode_strip(canvas, h - 60, height=44)


def render_processing(canvas, elapsed: float) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = L_BG
    cx, cy = w // 2, h // 2
    # 회전 spinner
    cv2.circle(canvas, (cx, cy), 50, L_DIVIDER, 6, cv2.LINE_AA)
    angle = int((elapsed * 360) % 360)
    cv2.ellipse(canvas, (cx, cy), (50, 50), -90, 0, max(60, angle // 2), L_GREEN_DARK, 8, cv2.LINE_AA)
    canvas[:] = kr(canvas, "분석 중...", (cx - kr_w("분석 중...", 24) // 2, cy + 80),
                  size=24, color=L_TEXT)
    canvas[:] = kr(canvas, "AI 코치가 점수와 피드백을 준비하고 있어요",
                  (cx - kr_w("AI 코치가 점수와 피드백을 준비하고 있어요", 14) // 2, cy + 120),
                  size=14, color=L_SUB)


def render_result(canvas, rubric, score_summary, feedback, replay_frame,
                  hover_label: Optional[str], progress: float, ctrl_boxes: Dict) -> None:
    h, w = canvas.shape[:2]
    pose_label = POSE_LABELS.get(rubric.pose_key, rubric.pose_key)
    draw_header(canvas, "실시간 자세 코칭", height=64)
    draw_workout_info_bar(canvas, pose_label, "측정 완료", y=80, height=52)
    # 좌측: 사용자 영상 리플레이 (피드백 카드 공간 확보 위해 320px로 줄임)
    main_y = 148
    main_h = 320
    cam_w_px = int(w * 0.62)
    draw_camera_frame(canvas, 16, main_y, cam_w_px, main_h, cam_bgr=replay_frame,
                     label="📹 REPLAY  내 자세 5초")
    # 우측: 결과 요약
    pxr = 16 + cam_w_px + 14
    pwr = w - 16 - pxr
    mean_score = int(score_summary.get("mean_score", 0))
    ox = score_summary.get("ox_accuracy", 0)
    verdict = score_summary.get("verdict", "-")
    draw_score_pill(canvas, pxr, main_y, pwr, 110, accuracy=mean_score,
                   label="가중 점수", remark=verdict)
    # 각도별 결과 (3개) — 작게
    dy = main_y + 124
    for k, spec in rubric.angles.items():
        acc = score_summary.get("per_angle_accuracy", {}).get(k, 0)
        diff = score_summary.get("per_angle_mean_diff", {}).get(k, 0)
        col = L_GREEN_DARK if acc >= 70 else L_AMBER
        draw_metric_row(canvas, pxr, dy, pwr, 60,
                       spec.name, f"{acc:.0f}% OK",
                       sub_label=f"평균 오차 {abs(diff):.1f}°")
        dy += 66

    # 친근체 피드백 (하단 카드) — 안전한 패딩 + 동적 높이 + 화면 공간 cap
    text = feedback.get("text", "") or ""
    fb_card_x1, fb_card_x2 = 16, w - 16    # 카드 좌우 외곽
    fb_inner_pad_x = 28                      # 카드 내부 좌우 패딩
    fb_pad_top = 12                          # 상단 패딩
    fb_pad_bot = 18                          # 하단 패딩
    fb_label_h = 32                          # "AI 코치" 라벨 영역
    fb_label_to_text_gap = 14
    fb_line_size = 15
    fb_line_h = 28                           # 한글 baseline 안전 (size+13)
    text_left_x = fb_card_x1 + fb_inner_pad_x
    text_right_x = fb_card_x2 - fb_inner_pad_x
    max_text_w = text_right_x - text_left_x

    from core.light_ui import kr_w as _kw
    from core.minimal_ui import wrap
    lines = wrap(text, max_text_w, fb_line_size)

    # 화면 공간 cap: 컨트롤 버튼(sy = h-120)과 모드 스트립(h-60) 안 침범
    button_top_y = h - 130                  # 컨트롤 버튼 시작 y (10px 여유 포함)
    fb_y = main_y + main_h + 14
    available = button_top_y - fb_y - 12    # 카드와 버튼 사이 12px 여유
    max_lines_by_space = max(1, (available - fb_pad_top - fb_label_h - fb_label_to_text_gap - fb_pad_bot) // fb_line_h)
    MAX_LINES = min(4, int(max_lines_by_space))
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        last = lines[-1]
        while _kw(last + " …", fb_line_size) > max_text_w and len(last) > 4:
            last = last[:-1]
        lines[-1] = last.rstrip() + " …"

    # 동적 높이
    fb_h = fb_pad_top + fb_label_h + fb_label_to_text_gap + len(lines) * fb_line_h + fb_pad_bot
    fb_h = max(fb_h, 86)
    fb_h = min(fb_h, available)             # 안전: 절대 버튼 침범 X

    soft_shadow_card(canvas, (fb_card_x1, fb_y), (fb_card_x2, fb_y + fb_h),
                      radius=14, shadow_offset=4)
    # AI 코치 라벨
    canvas[:] = kr(canvas, "💬 AI 코치", (text_left_x, fb_y + fb_pad_top),
                   size=15, color=L_GREEN_DARK)
    # 구분선
    sep_y = fb_y + fb_pad_top + fb_label_h - 2
    cv2.line(canvas, (text_left_x, sep_y), (text_right_x, sep_y), L_DIVIDER, 1)
    # 본문
    ty = sep_y + fb_label_to_text_gap
    for ln in lines:
        canvas[:] = kr(canvas, ln, (text_left_x, ty),
                       size=fb_line_size, color=L_TEXT)
        ty += fb_line_h

    # 액션 버튼들 (중앙 하단)
    bw = 220; bh = 48; gap = 16
    sx = (w - bw * 3 - gap * 2) // 2
    sy = h - 120
    for i, (lab, prim) in enumerate([("다시 시도", False), ("다른 자세", True), ("종료", False)]):
        x = sx + i * (bw + gap)
        ctrl_boxes[lab] = (x, sy, x + bw, sy + bh)
        active = (lab == hover_label)
        # primary = 강조 녹색
        if active:
            rounded_rect(canvas, (x, sy), (x + bw, sy + bh), L_GREEN_DARK, -1, bh // 2)
            tcol = (255, 255, 255)
        elif prim:
            rounded_rect(canvas, (x, sy), (x + bw, sy + bh), L_GREEN_BG, -1, bh // 2)
            rounded_rect(canvas, (x, sy), (x + bw, sy + bh), L_GREEN_DARK, 2, bh // 2)
            tcol = L_GREEN_DARK
        else:
            rounded_rect(canvas, (x, sy), (x + bw, sy + bh), L_CARD, -1, bh // 2)
            rounded_rect(canvas, (x, sy), (x + bw, sy + bh), (210, 210, 215), 2, bh // 2)
            tcol = L_TEXT
        tw = kr_w(lab, 17)
        canvas[:] = kr(canvas, lab, (x + (bw - tw) // 2, sy + (bh - 17) // 2 - 2),
                      size=17, color=tcol)
        if active and progress > 0:
            fw = int((bw - 16) * progress)
            cv2.rectangle(canvas, (x + 8, sy + bh - 6), (x + 8 + fw, sy + bh - 3),
                          (255, 255, 255) if hover_label == lab else L_GREEN_DARK, -1)
    # 모드 스트립
    draw_mode_strip(canvas, h - 60, height=44)


def estimate_alignment(landmarks, w: int, h: int) -> Dict:
    if landmarks is None:
        return {"score": 0.0, "msg": "전신이 보이게 카메라에서 떨어져 주세요"}
    head_v = getattr(landmarks[0], "visibility", 0.0)
    ankle_v = (getattr(landmarks[27], "visibility", 0.0) +
               getattr(landmarks[28], "visibility", 0.0)) / 2.0
    head_y = float(landmarks[0].y); ankle_y = max(float(landmarks[27].y), float(landmarks[28].y))
    in_frame = float(0 < head_y < 1) * float(0 < ankle_y < 1)
    coverage = ankle_y - head_y
    score = (head_v + ankle_v) / 2 * in_frame * min(1.0, coverage / 0.5)
    if score < 0.5:
        msg = "전신이 카메라에 다 보이도록 조정해 주세요"
    elif coverage < 0.4:
        msg = "조금만 더 떨어지면 좋아요"
    else:
        msg = "좋아요! 측정 준비 완료"
    return {"score": float(score), "msg": msg}


def save_session_report(pose_key, score_summary, feedback, latency):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": ts, "pose": pose_key, "score": score_summary,
              "feedback": feedback, "latency": latency}
    path = REPORTS_DIR / f"session_{ts}_{pose_key}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    hist = []
    if HIST_FILE.exists():
        try:
            hist = json.loads(HIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            hist = []
    rub = get_rubric(pose_key)
    hist.append({"timestamp": ts, "pose": pose_key, "pose_kr": rub.pose_name_kr,
                "score": score_summary["mean_score"], "ox": score_summary["ox_accuracy"],
                "verdict": score_summary["verdict"]})
    hist = hist[-50:]
    HIST_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] saved -> {path}")


def load_history(limit: int = 5):
    if not HIST_FILE.exists():
        return []
    try:
        return json.loads(HIST_FILE.read_text(encoding="utf-8"))[-limit:]
    except Exception:
        return []


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="OnPose Live Coach v6 (Light)")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--no-lifter", action="store_true")
    parser.add_argument("--no-bone-lock", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--no-distribution", action="store_true")
    parser.add_argument("--no-expert", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--no-splash", action="store_true")
    parser.add_argument("--variant", choices=["lite", "full", "heavy"], default="heavy")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--voice", action="store_true",
                        help="TTS 음성 (한국어 Heami만 있어 부자연스러울 수 있음)")
    parser.add_argument("--sound", action="store_true",
                        help="짧은 비프/차임벨 효과음 (권장)")
    parser.add_argument("--edge-tts", action="store_true",
                        help="Edge 신경망 TTS (인터넷 필요, 자연스러움)")
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--show-inpainted", action="store_true")
    args = parser.parse_args()

    if args.lite:
        args.variant = "lite"; args.no_lifter = True; args.offline = True

    if args.calibrated:
        if apply_calibrated_rubrics(THIS_DIR / "reports" / "rubric_calibrated.json"):
            print(f"[init] calibrated rubrics applied")

    use_distribution = not args.no_distribution
    distribution_rubrics = {}
    if use_distribution:
        distribution_rubrics = load_distribution_rubrics(THIS_DIR / "reports" / "pose_stats.json")
        if distribution_rubrics:
            print(f"[init] distribution-based scoring ON ({len(distribution_rubrics)} poses)")
        else:
            use_distribution = False

    runtime_mode = ("LITE" if args.lite else "OFFLINE" if args.offline
                    else f"STANDARD ({args.variant})")
    print("=" * 56); print(f"  OnPose v6 (Light)  |  {runtime_mode}"); print("=" * 56)

    profiler = LatencyProfiler()
    voice_mode = "tts" if args.voice else ("sound" if args.sound else "off")
    voice = VoiceCoach(mode=voice_mode, prefer_edge=args.edge_tts)
    if voice.enabled:
        print(f"[voice] mode = {voice_mode}  (edge_tts={voice.prefer_edge and voice._has_edge})")

    with profiler.measure("init_detector"):
        model_path = find_pose_landmarker(PROJECT_ROOT, args.variant)
        detector = build_detector(model_path, lite_mode=args.lite)
    with profiler.measure("init_lifter"):
        lifter = None if args.no_lifter else build_lifter()

    smoothing_on = not args.no_smooth
    pipeline = PoseAnglePipeline(detector, lifter, profiler=profiler,
                                  enable_bone_lock=not args.no_bone_lock,
                                  enable_smoothing=smoothing_on,
                                  enable_landmark_smooth=smoothing_on,
                                  enable_frame3d_smooth=smoothing_on,
                                  enable_occlusion_blend=smoothing_on)

    if args.video and args.video.exists():
        cap = cv2.VideoCapture(str(args.video))
        print(f"[input] video {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.record), fourcc, 20.0, (CANVAS_W, CANVAS_H))

    state = S_SPLASH if not args.no_splash else S_SELECT
    state_started_at = time.time()
    hold = HoldButton()
    selected_pose: Optional[str] = None
    rubric = None
    angle_history: List[Dict[str, float]] = []
    score_summary: Optional[Dict] = None
    feedback: Optional[Dict] = None
    expert_video: Optional[LoopVideo] = None
    recorder = FrameRecorder(max_frames=150, target_size=(640, 360))
    replay_idx = 0; last_replay_advance = 0.0
    history_for_chart = load_history(5)

    EXPERT_SIZE = (240, 160)
    print("[ready] q=quit  r=reset  1/2/3=quick pick  =:auto  s=screenshot")

    while cap.isOpened():
        frame_t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
        frame = cv2.flip(frame, 1)
        now = time.time()
        bgr, landmarks, angles, frame3d, meta = pipeline.process_frame(frame)
        h, w = bgr.shape[:2]

        # 라이트 캔버스 (배경 = 매우 밝은 회색)
        canvas = np.full((CANVAS_H, CANVAS_W, 3), L_BG, dtype=np.uint8)

        # hand pointer 좌표 변환
        scale_x = CANVAS_W / w; scale_y = CANVAS_H / h
        scaled_pointers = [(int(p[0] * scale_x), int(p[1] * scale_y))
                          for p in get_hand_pointers(landmarks, w, h)]

        if state == S_SPLASH:
            elapsed = now - state_started_at
            render_splash(canvas)
            if elapsed >= SPLASH_SECONDS:
                state = S_SELECT; state_started_at = now

        elif state == S_SELECT:
            boxes = render_select(canvas, landmarks, angles, None, 0.0, history_for_chart) or {}
            # hover 검출
            hover_key = None
            for k, b in boxes.items():
                for p in scaled_pointers:
                    if point_in(p, b):
                        hover_key = k; break
            progress, completed = hold.update(hover_key, now)
            # hover 강조 재렌더링 (active)
            if hover_key:
                canvas[:] = L_BG
                render_select(canvas, landmarks, angles, hover_key, progress, history_for_chart)
            # 포인터 표시
            for p in scaled_pointers:
                cv2.circle(canvas, p, 12, L_GREEN_DARK, 2, cv2.LINE_AA)
            if completed and hover_key is not None:
                selected_pose = hover_key
                rubric = distribution_rubrics.get(hover_key) or get_rubric(hover_key)
                state = S_INTRO; state_started_at = now
                angle_history = []; recorder.reset(); pipeline.reset_lifter()
                if expert_video is not None: expert_video.release()
                expert_video = None if args.no_expert else load_expert_video(hover_key, EXPERT_SIZE)
                hold.reset()
                voice.say_intro(rubric.pose_name_kr)
                print(f"[select] {hover_key}")

        elif state == S_INTRO:
            elapsed = now - state_started_at
            remain = max(0.0, INTRO_SECONDS - elapsed)
            alignment = estimate_alignment(landmarks, w, h)
            render_intro(canvas, rubric, remain, alignment)
            if elapsed >= INTRO_SECONDS:
                state = S_CAPTURE; state_started_at = now
                pipeline.reset_lifter(); recorder.reset()
                voice.say_start_capture()

        elif state == S_CAPTURE:
            elapsed = now - state_started_at
            remain = max(0.0, CAPTURE_SECONDS - elapsed)
            if landmarks is not None and any(angles.get(k, 0) > 0 for k in angles):
                angle_history.append(angles.copy())
            recorder.add(bgr)
            # 순간 점수
            if use_distribution and selected_pose in distribution_rubrics:
                per = score_frame_distribution(angles, distribution_rubrics[selected_pose])
            else:
                per = score_single_frame(angles, rubric)
            # 전문가 frame
            expert_frame = (expert_video.next_frame() if expert_video else None)
            render_capture(canvas, rubric, remain, angles, selected_pose,
                          cam_bgr=bgr, expert_frame=expert_frame,
                          captured=len(angle_history), score_summary_running=per,
                          landmarks=landmarks)
            if elapsed >= CAPTURE_SECONDS:
                state = S_PROCESSING; state_started_at = now
                voice.say_finish()

        elif state == S_PROCESSING:
            elapsed = now - state_started_at
            render_processing(canvas, elapsed)
            cv2.imshow("OnPose v6", canvas); cv2.waitKey(1)
            with profiler.measure("scoring"):
                if use_distribution and selected_pose in distribution_rubrics:
                    score_summary = score_sequence_distribution(angle_history,
                                                                 distribution_rubrics[selected_pose])
                else:
                    score_summary = score_sequence(angle_history, rubric)
            with profiler.measure("llm_feedback"):
                feedback = generate_feedback(rubric.pose_name_kr, score_summary,
                                              api_key=None if args.offline else GOOGLE_API_KEY,
                                              prefer_online=not args.offline)
            save_session_report(selected_pose, score_summary, feedback, profiler.to_dict())
            history_for_chart = load_history(5)
            voice.say_score(score_summary["mean_score"], score_summary["verdict"])
            state = S_RESULT; state_started_at = now
            replay_idx = 0; last_replay_advance = now

        elif state == S_RESULT:
            # 리플레이 frame
            replay_frame = None
            if not args.no_replay and recorder.count() > 0:
                if now - last_replay_advance > 0.05:
                    replay_idx = (replay_idx + 1) % recorder.count()
                    last_replay_advance = now
                replay_frame = recorder.get_loop_frame(replay_idx)
            ctrl_boxes: Dict[str, Tuple[int, int, int, int]] = {}
            render_result(canvas, rubric, score_summary, feedback, replay_frame,
                         hover_label=None, progress=0.0, ctrl_boxes=ctrl_boxes)
            # hover 검출
            hover_lbl = None
            for lab, b in ctrl_boxes.items():
                for p in scaled_pointers:
                    if point_in(p, b):
                        hover_lbl = lab; break
            progress, completed = hold.update(hover_lbl, now)
            if hover_lbl:
                # active 상태 재렌더
                ctrl_boxes2 = {}
                canvas[:] = L_BG
                render_result(canvas, rubric, score_summary, feedback, replay_frame,
                             hover_lbl, progress, ctrl_boxes2)
            for p in scaled_pointers:
                cv2.circle(canvas, p, 10, L_GREEN_DARK, 2, cv2.LINE_AA)
            if completed and hover_lbl == "다시 시도":
                state = S_INTRO; state_started_at = now
                angle_history = []; recorder.reset(); pipeline.reset_lifter()
                hold.reset()
            elif completed and hover_lbl == "다른 자세":
                state = S_SELECT; selected_pose = None; rubric = None
                if expert_video: expert_video.release(); expert_video = None
                recorder.reset(); hold.reset()
            elif completed and hover_lbl == "종료":
                break

        cv2.imshow("OnPose v6", canvas)
        if writer: writer.write(canvas)
        profiler.record("frame_total", (time.perf_counter() - frame_t0) * 1000.0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("r"):
            state = S_SELECT; selected_pose = None; rubric = None
            if expert_video: expert_video.release(); expert_video = None
            recorder.reset(); hold.reset()
        elif key in (ord("="), ord("+")) and state == S_SELECT and landmarks is not None:
            auto_key, conf, _ = classify_pose(angles)
            if auto_key and conf > 0.4:
                selected_pose = auto_key
                rubric = distribution_rubrics.get(auto_key) or get_rubric(auto_key)
                state = S_INTRO; state_started_at = now
                angle_history = []; recorder.reset(); pipeline.reset_lifter()
                if expert_video: expert_video.release()
                expert_video = None if args.no_expert else load_expert_video(auto_key, EXPERT_SIZE)
                hold.reset(); voice.say_intro(rubric.pose_name_kr)
        elif key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if 0 <= idx < len(POSE_ORDER):
                selected_pose = POSE_ORDER[idx]
                rubric = distribution_rubrics.get(selected_pose) or get_rubric(selected_pose)
                state = S_INTRO; state_started_at = now
                angle_history = []; recorder.reset(); pipeline.reset_lifter()
                if expert_video: expert_video.release()
                expert_video = None if args.no_expert else load_expert_video(selected_pose, EXPERT_SIZE)
                hold.reset(); voice.say_intro(rubric.pose_name_kr)
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(REPORTS_DIR / f"screenshot_{ts}.png"), canvas)

    cap.release()
    if expert_video: expert_video.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    profiler.report("Final Latency Report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
