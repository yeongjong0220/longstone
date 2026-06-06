"""
minmin 스타일을 발전시킨 미니멀 UI 모듈.

핵심 원칙:
  1. 카메라가 메인 (전체 화면), UI는 가벼운 오버레이
  2. 단색 + 라운드 카드 (그라데이션 자제)
  3. 큰 숫자/짧은 단어 (한 눈에 인식)
  4. 색 코드 일관: GREEN=PASS, RED=WARN, CYAN=INFO, GOLD=ACCENT
  5. 두꺼운 한글 폰트 + 명확한 대비
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgunbd.ttf"),       # 굵은 버전 우선
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]
KR_FONT = next((p for p in _FONT_CANDIDATES if p.exists()), None)

# Minimal palette (BGR)
M_DARK = (28, 28, 38)
M_CARD = (44, 44, 58)
M_CARD_HI = (60, 60, 80)
M_BORDER = (90, 90, 120)
M_TEXT = (240, 240, 245)
M_DIM = (155, 155, 175)
M_GOLD = (60, 200, 255)        # accent
M_GREEN = (100, 230, 130)
M_RED = (95, 95, 240)
M_CYAN = (215, 200, 90)
M_PURPLE = (220, 130, 180)


def kr(img, text, xy, size=18, color=M_TEXT, bold=True, stroke=0):
    """한글 텍스트 — 굵은 폰트 + 옵션 stroke 으로 가독성 강화"""
    if _PIL and KR_FONT is not None:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        font = ImageFont.truetype(str(KR_FONT), size)
        fill = (color[2], color[1], color[0])
        if stroke > 0:
            d.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))
        else:
            d.text(xy, text, font=font, fill=fill)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, max(size / 36.0, 0.5), color, 2)
    return img


def kr_w(text, size):
    if _PIL and KR_FONT is not None:
        font = ImageFont.truetype(str(KR_FONT), size)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    return int(len(text) * size * 0.6)


def rounded_rect(img, p1, p2, color, thickness=-1, radius=10):
    x1, y1 = p1; x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2 - 1, (y2 - y1) // 2 - 1))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    return img


def overlay_card(img, p1, p2, alpha: float = 0.85, color=M_CARD, border=M_BORDER, radius=12) -> np.ndarray:
    """반투명 라운드 카드 — 카메라 위 정보 오버레이용. ROI 영역만 합성해서 in-place 보장."""
    x1, y1 = p1; x2, y2 = p2
    H, W = img.shape[:2]
    rx1 = max(0, x1); ry1 = max(0, y1)
    rx2 = min(W, x2); ry2 = min(H, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return img
    roi = img[ry1:ry2, rx1:rx2].copy()
    overlay = np.zeros_like(roi)
    # 라운드 마스크
    r = max(1, min(radius, (rx2 - rx1) // 2 - 1, (ry2 - ry1) // 2 - 1))
    h_m = ry2 - ry1; w_m = rx2 - rx1
    mask = np.zeros((h_m, w_m), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w_m - r, h_m), 255, -1)
    cv2.rectangle(mask, (0, r), (w_m, h_m - r), 255, -1)
    for cx, cy in ((r, r), (w_m - r, r), (r, h_m - r), (w_m - r, h_m - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    overlay[:] = color
    m3 = (mask[..., None] / 255.0) * alpha
    img[ry1:ry2, rx1:rx2] = (overlay * m3 + roi * (1 - m3)).astype(np.uint8)
    # 테두리
    rounded_rect(img, p1, p2, border, thickness=1, radius=radius)
    return img


def status_chip(img, text: str, xy, color=M_GREEN, pad: int = 10) -> np.ndarray:
    """PASS / WARN 같은 작은 상태 칩"""
    x, y = xy
    size = 18
    w = kr_w(text, size) + pad * 2
    h = size + pad
    img = rounded_rect(img, (x, y), (x + w, y + h), color, thickness=-1, radius=h // 2)
    img = kr(img, text, (x + pad, y + 3), size=size, color=(20, 20, 30))
    return img, w


def vertical_bar(img, x: int, y: int, h: int, value: float, max_value: float = 100,
                color=M_GREEN, bg=M_CARD_HI, width: int = 8) -> np.ndarray:
    """세로 막대 — 점수/정확도 표시"""
    rounded_rect(img, (x, y), (x + width, y + h), bg, thickness=-1, radius=width // 2)
    fill_h = int(h * max(0.0, min(1.0, value / max_value)))
    if fill_h > 0:
        rounded_rect(img, (x, y + h - fill_h), (x + width, y + h), color,
                     thickness=-1, radius=width // 2)
    return img


def color_for_score(score: float) -> Tuple[int, int, int]:
    if score >= 85: return M_GREEN
    if score >= 70: return M_GOLD
    if score >= 55: return M_CYAN
    return M_RED


# -----------------------------------------------------------------------------
# 미니멀 골격 (가중치 시각화)
# -----------------------------------------------------------------------------
def draw_minimal_skeleton(img, landmarks, pose_key: Optional[str] = None,
                          highlight: Optional[List[int]] = None,
                          inpainted_flags: Optional[Dict[int, bool]] = None) -> np.ndarray:
    """미니멀 골격 + critical 강조 + in-paint된 키포인트 노란색 점선으로 표시.

    Args:
        landmarks: MediaPipe 33-keypoint landmarks
        pose_key: 자세별 색상
        highlight: critical 관절 인덱스
        inpainted_flags: {idx: True/False} — 가려져서 추정된 관절은 다르게 그림
    """
    if landmarks is None:
        return img
    h, w = img.shape[:2]
    BONES_33 = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    KEY_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    base_color = (240, 220, 100) if pose_key == "The_Seal" else \
                 (130, 230, 200) if pose_key == "Bridging" else \
                 (180, 170, 240)
    inpaint_color = (80, 200, 255)   # 노란빛 (가려진 → 추정된 키포인트)
    highlight_color = (90, 230, 130)
    pts = {}; pts_inpainted = {}
    for idx in KEY_POINTS:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", 1.0)
        is_inp = inpainted_flags.get(idx, False) if inpainted_flags else False
        # in-paint 표시할 때는 visibility 무관, 아니면 0.30 이상만
        if not is_inp and vis < 0.30:
            continue
        pts[idx] = (int(lm.x * w), int(lm.y * h))
        if is_inp:
            pts_inpainted[idx] = True
    # 본 (양 끝 중 하나라도 inpainted면 점선)
    for a, b in BONES_33:
        if a in pts and b in pts:
            is_inp = pts_inpainted.get(a, False) or pts_inpainted.get(b, False)
            if is_inp:
                # 점선
                _draw_dashed_line(img, pts[a], pts[b], inpaint_color, thickness=2, dash=8)
            else:
                cv2.line(img, pts[a], pts[b], base_color, 3, cv2.LINE_AA)
    # 점
    for idx, p in pts.items():
        is_inp = pts_inpainted.get(idx, False)
        if is_inp:
            # 가려진 키포인트 — 빈 원 (테두리만)
            cv2.circle(img, p, 7, inpaint_color, 2, cv2.LINE_AA)
            cv2.circle(img, p, 3, inpaint_color, -1, cv2.LINE_AA)
        elif highlight and idx in highlight:
            cv2.circle(img, p, 8, highlight_color, -1, cv2.LINE_AA)
            cv2.circle(img, p, 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 5, base_color, -1, cv2.LINE_AA)
    return img


def _draw_dashed_line(img, p1, p2, color, thickness=2, dash=8) -> None:
    x1, y1 = p1; x2, y2 = p2
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if dist < 1:
        return
    steps = max(1, int(dist // dash))
    for i in range(steps):
        if i % 2 != 0:
            continue
        a = i / steps; b = (i + 1) / steps
        sx, sy = int(x1 + a * (x2 - x1)), int(y1 + a * (y2 - y1))
        ex, ey = int(x1 + b * (x2 - x1)), int(y1 + b * (y2 - y1))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# 상단 헤더 바 / 하단 피드백 바 / 좌측 정보 카드
# -----------------------------------------------------------------------------
def draw_top_bar(img, title: str, mode: str = "STANDARD",
                 state_text: str = "", state_color=M_CYAN, height: int = 56) -> np.ndarray:
    """상단 헤더 — in-place 동기화"""
    h, w = img.shape[:2]
    # 헤더 영역만 ROI 다크닝
    roi = img[:height].copy()
    bar = np.full_like(roi, (15, 15, 22))
    img[:height] = cv2.addWeighted(bar, 0.85, roi, 0.15, 0)
    cv2.line(img, (0, height), (w, height), M_BORDER, 1)
    new = kr(img, title, (16, 13), size=22, color=M_TEXT)
    img[:] = new
    mw = kr_w(mode, 14) + 18
    mx = 16 + kr_w(title, 22) + 18
    rounded_rect(img, (mx, 17), (mx + mw, 17 + 26), M_CARD_HI, -1, 13)
    new = kr(img, mode, (mx + 9, 19), size=14, color=M_GOLD)
    img[:] = new
    if state_text:
        sw = kr_w(state_text, 18) + 30
        sx = w - sw - 16
        rounded_rect(img, (sx, 12), (sx + sw, 12 + 32), M_CARD, -1, 16)
        cv2.circle(img, (sx + 14, 28), 5, state_color, -1)
        new = kr(img, state_text, (sx + 26, 17), size=16, color=M_TEXT)
        img[:] = new
    return img


def draw_bottom_feedback(img, text: str, color=M_GOLD, height: int = 60) -> np.ndarray:
    """하단 피드백 바 — in-place"""
    h, w = img.shape[:2]
    y = h - height
    roi = img[y:].copy()
    bar = np.full_like(roi, (15, 15, 22))
    img[y:] = cv2.addWeighted(bar, 0.88, roi, 0.12, 0)
    cv2.line(img, (0, y), (w, y), M_BORDER, 1)
    cv2.rectangle(img, (0, y), (5, h), color, -1)
    cx = 24; cy = y + height // 2
    cv2.circle(img, (cx, cy), 14, color, -1)
    new = kr(img, "AI", (cx - 13, cy - 11), size=14, color=(20, 20, 30))
    img[:] = new
    new = kr(img, text, (50, y + 18), size=18, color=M_TEXT)
    img[:] = new
    return img


def draw_left_info_card(img, lines: List[Tuple[str, str, Tuple[int, int, int]]],
                        x: int = 16, y: int = 76, width: int = 280) -> np.ndarray:
    """좌측 정보 카드 — in-place"""
    line_h = 56
    height = 20 + len(lines) * line_h
    overlay_card(img, (x, y), (x + width, y + height), alpha=0.86)   # in-place
    for i, (label, value, color) in enumerate(lines):
        ly = y + 14 + i * line_h
        new = kr(img, label, (x + 16, ly), size=13, color=M_DIM)
        img[:] = new
        new = kr(img, value, (x + 16, ly + 18), size=28, color=color)
        img[:] = new
    return img


def draw_pose_card(img, pose_kr: str, pose_key: str,
                  x: int, y: int, w: int = 200, h: int = 160,
                  hover_progress: float = 0.0, active: bool = False,
                  selected: bool = False) -> np.ndarray:
    """자세 선택 카드 — 미니멀 + 자세 일러스트.
    카메라 위에 떠 있는 카드 → 부분 그림자 + 진한 배경 + 강한 테두리.
    """
    H_IMG, W_IMG = img.shape[:2]
    # 1) 그림자: 카드 영역 + offset에만 ROI 다크닝
    so_x1 = max(0, x + 6); so_y1 = max(0, y + 8)
    so_x2 = min(W_IMG, x + w + 6); so_y2 = min(H_IMG, y + h + 8)
    if so_x2 > so_x1 and so_y2 > so_y1:
        roi = img[so_y1:so_y2, so_x1:so_x2]
        dark = np.zeros_like(roi)
        # 라운드 마스크
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        r = 16
        h_m, w_m = mask.shape
        cv2.rectangle(mask, (r, 0), (w_m - r, h_m), 255, -1)
        cv2.rectangle(mask, (0, r), (w_m, h_m - r), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1)
        cv2.circle(mask, (w_m - r, r), r, 255, -1)
        cv2.circle(mask, (r, h_m - r), r, 255, -1)
        cv2.circle(mask, (w_m - r, h_m - r), r, 255, -1)
        m3 = (mask[..., None] / 255.0) * 0.55
        img[so_y1:so_y2, so_x1:so_x2] = (roi * (1 - m3) + dark * m3).astype(np.uint8)
    # 2) 카드 배경 (완전 불투명)
    if active:
        bg = (50, 95, 60)
        border = M_GREEN
        border_thick = 3
    elif selected:
        bg = (60, 75, 95)
        border = M_GOLD
        border_thick = 2
    else:
        bg = (40, 40, 56)
        border = (130, 130, 160)
        border_thick = 2
    rounded_rect(img, (x, y), (x + w, y + h), bg, -1, 16)
    rounded_rect(img, (x, y), (x + w, y + h), border, border_thick, 16)
    # 3) 자세 아이콘 (상단 60%)
    icon_h = int(h * 0.60)
    icon_color = (190, 255, 230) if active else (M_GOLD if selected else (215, 225, 245))
    _draw_pose_icon_minimal(img, x + 16, y + 16, x + w - 16, y + icon_h, pose_key, color=icon_color)
    # 4) 라벨 배지 (하단 — 진한 배경 + 굵은 흰 글자)
    badge_y1 = y + icon_h + 4
    badge_y2 = y + h - 16
    badge_color = (28, 60, 38) if active else (22, 22, 32)
    rounded_rect(img, (x + 14, badge_y1), (x + w - 14, badge_y2), badge_color, -1, 8)
    if active:
        rounded_rect(img, (x + 14, badge_y1), (x + w - 14, badge_y2), M_GREEN, 1, 8)
    label_size = 26
    tw = kr_w(pose_kr, label_size)
    label_color = (255, 255, 255)
    badge_h = badge_y2 - badge_y1
    text_y = badge_y1 + (badge_h - label_size) // 2 - 6
    # in-place 동기화: kr 결과를 원본 배열에 복사 (호출자가 return값 안 받아도 그림 유지)
    new_img = kr(img, pose_kr, (x + (w - tw) // 2, text_y), size=label_size, color=label_color)
    img[:] = new_img
    # 5) 진행 바 (카드 최하단)
    if hover_progress > 0:
        fw = int((w - 24) * hover_progress)
        rounded_rect(img, (x + 12, y + h - 10), (x + 12 + fw, y + h - 6), M_GREEN, -1, 2)
    return img


def _draw_pose_icon_minimal(img, x1, y1, x2, y2, pose_key, color=M_GOLD):
    """자세별 미니멀 아이콘"""
    bw, bh = x2 - x1, y2 - y1
    icons = {
        "The_Seal": {
            "j": {"head": (.50, .25), "neck": (.50, .38), "sh": (.55, .43),
                  "hip": (.52, .65), "knee": (.40, .55), "ankle": (.36, .48),
                  "hand": (.45, .55)},
            "b": [("head", "neck"), ("neck", "sh"), ("sh", "hip"), ("hip", "knee"),
                  ("knee", "ankle"), ("sh", "hand")],
        },
        "Spine_Stretch": {
            "j": {"head": (.38, .35), "neck": (.45, .42), "sh": (.52, .47),
                  "hip": (.60, .68), "knee": (.78, .68), "ankle": (.92, .68),
                  "hand": (.28, .50)},
            "b": [("head", "neck"), ("neck", "sh"), ("sh", "hip"), ("hip", "knee"),
                  ("knee", "ankle"), ("sh", "hand")],
        },
        "Bridging": {
            "j": {"head": (.18, .70), "neck": (.26, .65), "sh": (.34, .58),
                  "hip": (.58, .42), "knee": (.74, .54), "ankle": (.80, .72)},
            "b": [("head", "neck"), ("neck", "sh"), ("sh", "hip"), ("hip", "knee"),
                  ("knee", "ankle")],
        },
    }
    cfg = icons.get(pose_key)
    if cfg is None:
        return img
    pts = {n: (x1 + int(nx * bw), y1 + int(ny * bh)) for n, (nx, ny) in cfg["j"].items()}
    for a, b in cfg["b"]:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)
    return img


def draw_circular_progress(img, cx: int, cy: int, radius: int, value: float, max_v: float = 100,
                           label: str = "", color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """원형 게이지 — in-place"""
    if color is None:
        color = color_for_score(value if max_v == 100 else value / max_v * 100)
    cv2.circle(img, (cx, cy), radius, M_CARD_HI, 6, cv2.LINE_AA)
    angle = int(360 * max(0.0, min(1.0, value / max_v)))
    if angle > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, angle, color, 8, cv2.LINE_AA)
    txt = f"{int(value)}"
    tw = kr_w(txt, 36)
    new = kr(img, txt, (cx - tw // 2, cy - 22), size=36, color=M_TEXT)
    img[:] = new
    if label:
        lw = kr_w(label, 14)
        new = kr(img, label, (cx - lw // 2, cy + 18), size=14, color=M_DIM)
        img[:] = new
    return img


def draw_angle_dial(img, x: int, y: int, name: str, value: float, target: float,
                   tolerance: float, w: int = 240, h: int = 56) -> np.ndarray:
    """각도 + 정답 + 허용범위 한 줄 다이얼. 항상 in-place 보장."""
    diff = abs(value - target)
    ok = diff <= tolerance
    color = M_GREEN if ok else M_RED
    # 카드 (in-place로 작동)
    new = overlay_card(img, (x, y), (x + w, y + h), alpha=0.85, color=M_CARD, border=M_BORDER, radius=10)
    img[:] = new
    # 텍스트 4개를 한 번에 PIL로 그려서 in-place로 복사
    new = kr(img, name, (x + 12, y + 6), size=13, color=M_DIM)
    new = kr(new, f"{value:.0f}°", (x + 12, y + 22), size=24, color=color)
    new = kr(new, f"목표 {target:.0f}° (±{tolerance:.0f})", (x + 110, y + 8), size=12, color=M_DIM)
    img[:] = new
    # 미니 바: 오차/허용범위 비율
    bx = x + 110; by = y + 30; bw = w - 110 - 16; bh = 6
    rounded_rect(img, (bx, by), (bx + bw, by + bh), M_CARD_HI, -1, 3)
    rel = min(1.0, diff / (tolerance * 2))
    rounded_rect(img, (bx, by), (bx + int(bw * rel), by + bh), color, -1, 3)
    # OK/X 표시
    if ok:
        cv2.circle(img, (x + w - 20, y + 22), 9, M_GREEN, -1)
        new = kr(img, "O", (x + w - 26, y + 11), size=18, color=(20, 20, 30))
    else:
        cv2.circle(img, (x + w - 20, y + 22), 9, M_RED, -1)
        new = kr(img, "X", (x + w - 25, y + 11), size=18, color=(20, 20, 30))
    img[:] = new
    return img


def wrap(text: str, max_w: int, size: int) -> List[str]:
    if not text:
        return []
    words = text.split(" ")
    out = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if kr_w(cand, size) <= max_w:
            cur = cand
        else:
            if cur:
                out.append(cur)
            if kr_w(w, size) > max_w:
                buf = ""
                for ch in w:
                    if kr_w(buf + ch, size) <= max_w:
                        buf += ch
                    else:
                        out.append(buf); buf = ch
                cur = buf
            else:
                cur = w
    if cur: out.append(cur)
    return out
