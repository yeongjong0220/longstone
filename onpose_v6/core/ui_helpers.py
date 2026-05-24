"""
v6 UI 공통 헬퍼 — 한글 폰트, 패널, 점수 게이지, 3분할 합성.
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
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]
KR_FONT = next((p for p in _FONT_CANDIDATES if p.exists()), None)

COLOR_BG = (24, 24, 32)
COLOR_PANEL = (38, 38, 50)
COLOR_PANEL_BORDER = (60, 180, 220)
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (170, 170, 180)
COLOR_OK = (80, 220, 140)
COLOR_WARN = (60, 200, 240)
COLOR_BAD = (90, 110, 240)

# 모던 그라데이션 팔레트
COLOR_BTN_BG_TOP = (52, 52, 72)
COLOR_BTN_BG_BOT = (32, 32, 48)
COLOR_BTN_HOVER_TOP = (60, 130, 90)
COLOR_BTN_HOVER_BOT = (35, 90, 60)
COLOR_BTN_BORDER = (110, 110, 140)
COLOR_BTN_BORDER_HOVER = (90, 230, 160)
COLOR_ACCENT = (130, 200, 255)


def draw_kr(img: np.ndarray, text: str, xy: Tuple[int, int],
            size: int = 24, color: Tuple[int, int, int] = COLOR_TEXT) -> np.ndarray:
    if _PIL and KR_FONT is not None:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        font = ImageFont.truetype(str(KR_FONT), size)
        d.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, max(size / 36.0, 0.5), color, 2)
    return img


def measure_kr(text: str, size: int) -> int:
    if _PIL and KR_FONT is not None:
        font = ImageFont.truetype(str(KR_FONT), size)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    return int(len(text) * size * 0.6)


def draw_rounded_rect(img: np.ndarray, pt1, pt2, color, thickness: int = -1, radius: int = 12) -> np.ndarray:
    x1, y1 = pt1; x2, y2 = pt2
    r = max(1, min(radius, (x2 - x1) // 2 - 1, (y2 - y1) // 2 - 1))
    if thickness < 0:  # filled
        # 중앙 직사각형 두 개
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        # 네 모서리 원
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
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


def draw_gradient_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       top_color, bottom_color, radius: int = 14) -> np.ndarray:
    """그라데이션 + 둥근 모서리. 일단 임시 ROI에 그라데이션 그리고 라운드 마스크 합성."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    tc = np.array(top_color, dtype=np.float32)
    bc = np.array(bottom_color, dtype=np.float32)
    for yy in range(h):
        t = yy / max(1, h - 1)
        grad[yy, :] = (tc * (1 - t) + bc * t).astype(np.uint8)
    # 라운드 마스크
    mask = np.zeros((h, w), dtype=np.uint8)
    r = max(1, min(radius, w // 2 - 1, h // 2 - 1))
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    roi = img[y1:y1 + h, x1:x1 + w]
    if roi.shape[:2] != grad.shape[:2]:
        return img
    mask3 = mask[..., None] / 255.0
    blended = (grad * mask3 + roi * (1 - mask3)).astype(np.uint8)
    img[y1:y1 + h, x1:x1 + w] = blended
    return img


def draw_drop_shadow(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     offset: int = 4, radius: int = 14, alpha: float = 0.30) -> np.ndarray:
    """둥근 직사각형 그림자 — 박스 입체감"""
    overlay = img.copy()
    draw_rounded_rect(overlay, (x1 + offset, y1 + offset), (x2 + offset, y2 + offset),
                      (0, 0, 0), thickness=-1, radius=radius)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# 자세별 상징 아이콘 (단순 골격 점/선)
POSE_ICONS = {
    "The_Seal": {
        "joints": {"head": (0.50, 0.30), "neck": (0.50, 0.40), "shoulder": (0.55, 0.45),
                   "hip": (0.50, 0.62), "knee": (0.40, 0.55), "ankle": (0.36, 0.50),
                   "hand": (0.45, 0.52)},
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hand")],
    },
    "Spine_Stretch": {
        "joints": {"head": (0.42, 0.40), "neck": (0.47, 0.45), "shoulder": (0.52, 0.48),
                   "hip": (0.62, 0.62), "knee": (0.78, 0.62), "ankle": (0.92, 0.62),
                   "hand": (0.32, 0.50)},
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hand")],
    },
    "Bridging": {
        "joints": {"head": (0.20, 0.70), "neck": (0.28, 0.65), "shoulder": (0.36, 0.60),
                   "hip": (0.58, 0.45), "knee": (0.72, 0.55), "ankle": (0.78, 0.72)},
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle")],
    },
}


def draw_pose_icon(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   pose_key: str, color=(255, 255, 255)) -> np.ndarray:
    """박스 내부에 작은 자세 일러스트 그리기"""
    cfg = POSE_ICONS.get(pose_key)
    if cfg is None:
        return img
    bw, bh = x2 - x1, y2 - y1
    pts = {name: (x1 + int(nx * bw), y1 + int(ny * bh)) for name, (nx, ny) in cfg["joints"].items()}
    for a, b in cfg["bones"]:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
    return img


def draw_filled_panel(img: np.ndarray, x: int, y: int, w: int, h: int,
                      title: str = "", border_color: Tuple[int, int, int] = COLOR_PANEL_BORDER) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL, -1)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)
    if title:
        img = draw_kr(img, title, (x + 14, y + 8), size=18, color=COLOR_WARN)
    return img


def draw_score_gauge(img: np.ndarray, x: int, y: int, w: int, h: int, score: float,
                     label: str = "점수") -> np.ndarray:
    """가로 게이지 — 점수에 따라 색 변경"""
    if score >= 85:
        color = COLOR_OK
    elif score >= 70:
        color = COLOR_WARN
    else:
        color = COLOR_BAD
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 75), -1)
    fill_w = int(w * max(0.0, min(1.0, score / 100.0)))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 210), 1)
    img = draw_kr(img, f"{label}  {score:.1f}/100", (x + 8, y - 24), size=18, color=COLOR_TEXT)
    return img


def draw_horizontal_bars(img: np.ndarray, x: int, y: int, w: int,
                        rows: List[Tuple[str, float, bool]],
                        row_h: int = 30, gap: int = 8) -> np.ndarray:
    """각도별 정확도 가로 막대.  rows = [(label, percent_0_to_100, ok_flag), ...]"""
    for i, (label, pct, ok) in enumerate(rows):
        yy = y + i * (row_h + gap)
        img = draw_kr(img, label, (x, yy), size=16, color=COLOR_TEXT)
        bar_x = x + 110
        bar_w = w - 110 - 60
        cv2.rectangle(img, (bar_x, yy + 4), (bar_x + bar_w, yy + 4 + row_h - 8), (55, 55, 70), -1)
        fill = int(bar_w * max(0.0, min(1.0, pct / 100.0)))
        color = COLOR_OK if ok else COLOR_BAD
        cv2.rectangle(img, (bar_x, yy + 4), (bar_x + fill, yy + 4 + row_h - 8), color, -1)
        img = draw_kr(img, f"{pct:.0f}%", (bar_x + bar_w + 10, yy), size=16, color=COLOR_TEXT)
    return img


def draw_centered_message(img: np.ndarray, lines: List[Tuple[str, int]],
                          color: Tuple[int, int, int] = COLOR_TEXT) -> np.ndarray:
    h, w = img.shape[:2]
    panel_h = sum(s + 16 for _, s in lines) + 60
    panel_w = max(int(w * 0.55), max((measure_kr(t, s) for t, s in lines), default=0) + 100)
    x = (w - panel_w) // 2
    y = (h - panel_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), COLOR_PANEL_BORDER, 2)
    cur_y = y + 28
    for text, size in lines:
        tw = measure_kr(text, size)
        tx = x + (panel_w - tw) // 2
        img = draw_kr(img, text, (tx, cur_y), size=size, color=color)
        cur_y += size + 16
    return img


def draw_countdown_ring(img: np.ndarray, cx: int, cy: int, radius: int,
                       remain: float, total: float, label: str = "") -> np.ndarray:
    """원형 카운트다운 — 남은 시간 시각화"""
    cv2.circle(img, (cx, cy), radius, (60, 60, 75), 4)
    angle = int(360 * (1.0 - max(0.0, remain) / max(total, 0.001)))
    if angle > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, angle, COLOR_OK, 6)
    img = draw_kr(img, f"{max(0.0, remain):.1f}s", (cx - 30, cy - 18), size=30, color=COLOR_TEXT)
    if label:
        img = draw_kr(img, label, (cx - measure_kr(label, 18) // 2, cy + 26), size=18, color=COLOR_DIM)
    return img


def compose_layout(main_bgr: np.ndarray, side_bgr: Optional[np.ndarray],
                  panel_bgr: np.ndarray, canvas_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
    """
    3분할: [좌 큰 카메라] + [우상 참고영상] + [우하 점수/피드백 패널]
    main_bgr  : 사용자 웹캠
    side_bgr  : 참고 영상 프레임 (None이면 공백)
    panel_bgr : 점수/피드백 패널
    """
    W, H = canvas_size
    canvas = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
    # 좌측 camera = 60% 너비
    cam_w = int(W * 0.62)
    cam_h = H
    cam = cv2.resize(main_bgr, (cam_w, cam_h))
    canvas[0:cam_h, 0:cam_w] = cam

    side_x = cam_w
    side_y = 0
    side_w = W - cam_w
    side_h = int(H * 0.42)
    if side_bgr is not None:
        side = cv2.resize(side_bgr, (side_w, side_h))
        canvas[side_y:side_y + side_h, side_x:side_x + side_w] = side
    else:
        cv2.rectangle(canvas, (side_x, side_y), (side_x + side_w, side_y + side_h), (40, 40, 55), -1)
        canvas = draw_kr(canvas, "전문가 참고 영상", (side_x + 16, side_y + 14), size=18, color=COLOR_WARN)
        canvas = draw_kr(canvas, "(영상 자리)", (side_x + 16, side_y + 60), size=22, color=COLOR_DIM)

    # 우하 panel
    pan_y = side_h
    pan_h = H - side_h
    panel_resized = cv2.resize(panel_bgr, (side_w, pan_h))
    canvas[pan_y:pan_y + pan_h, side_x:side_x + side_w] = panel_resized

    cv2.rectangle(canvas, (side_x, side_y), (side_x + side_w - 1, side_y + side_h - 1), COLOR_PANEL_BORDER, 2)
    cv2.rectangle(canvas, (side_x, pan_y), (side_x + side_w - 1, pan_y + pan_h - 1), COLOR_PANEL_BORDER, 2)
    return canvas


def make_panel_image(size: Tuple[int, int] = (480, 420)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), COLOR_PANEL, dtype=np.uint8)
    return img


def wrap_text(text: str, max_width_px: int, size: int) -> List[str]:
    """한국어 친화 단순 줄바꿈 — 글자 단위 또는 공백 단위."""
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if measure_kr(candidate, size) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            # 단일 단어가 폭 초과 → 글자 단위 분할
            if measure_kr(w, size) > max_width_px:
                buf = ""
                for ch in w:
                    if measure_kr(buf + ch, size) <= max_width_px:
                        buf += ch
                    else:
                        lines.append(buf)
                        buf = ch
                cur = buf
            else:
                cur = w
    if cur:
        lines.append(cur)
    return lines
