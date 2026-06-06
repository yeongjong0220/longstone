"""
NCCOSS 참고 디자인 기반 밝은 테마 UI.

- 배경: 매우 밝은 회색 (#F5F5F7)
- 카드: 흰색 + 부드러운 그림자
- 텍스트: 진한 검정 (#1A1A1A) + 보조 회색 (#6B7280)
- 액센트: 산뜻한 녹색 (#3DDC84, #16A34A)
- 경고: 노랑 (#F59E0B), 정보: 파랑 (#3B82F6)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
]
KR_FONT = next((p for p in _FONT_CANDIDATES if p.exists()), None)

# ======= Light Palette (BGR) =======
L_BG = (247, 245, 245)           # #F5F5F7 - 메인 배경
L_CARD = (255, 255, 255)         # #FFFFFF - 카드
L_TEXT = (28, 28, 28)            # #1C1C1C - 메인 텍스트
L_SUB = (115, 113, 107)          # #6B7173 - 보조 텍스트
L_DIVIDER = (232, 229, 229)      # 구분선
L_GREEN_BG = (231, 250, 235)     # 살짝 녹색 배경
L_GREEN = (132, 220, 80)         # #50DC84
L_GREEN_DARK = (74, 173, 60)     # #3CAD4A 강조
L_AMBER = (24, 156, 245)         # #F59C18 BGR
L_AMBER_BG = (220, 240, 250)     # 살짝 노랑 배경
L_BLUE = (235, 130, 59)          # #3B82EB BGR
L_BLUE_BG = (250, 240, 230)
L_RED = (95, 71, 232)            # 빨강
L_RED_BG = (235, 226, 248)
L_HEADER_BG = (252, 252, 252)


def kr(img, text, xy, size=18, color=L_TEXT, stroke=0):
    if _PIL and KR_FONT is not None:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        font = ImageFont.truetype(str(KR_FONT), size)
        fill = (color[2], color[1], color[0])
        if stroke > 0:
            d.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=(255, 255, 255))
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


def rounded_rect(img, p1, p2, color, thickness=-1, radius=12):
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


def soft_shadow_card(img, p1, p2, radius=16, shadow_offset=6, shadow_alpha=0.10) -> None:
    """부드러운 그림자 + 흰 카드 (in-place)."""
    x1, y1 = p1; x2, y2 = p2
    H, W = img.shape[:2]
    # 그림자 — 카드 아래쪽으로 살짝 떨어진 영역 다크닝
    sy1 = max(0, y1 + shadow_offset); sy2 = min(H, y2 + shadow_offset + 4)
    sx1 = max(0, x1 + shadow_offset // 2); sx2 = min(W, x2 + shadow_offset // 2)
    if sx2 > sx1 and sy2 > sy1:
        roi = img[sy1:sy2, sx1:sx2]
        dark = np.full_like(roi, (220, 220, 220))
        # 라운드 마스크
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        r = max(1, min(radius, (sx2 - sx1) // 2 - 1, (sy2 - sy1) // 2 - 1))
        h_m, w_m = mask.shape
        cv2.rectangle(mask, (r, 0), (w_m - r, h_m), 255, -1)
        cv2.rectangle(mask, (0, r), (w_m, h_m - r), 255, -1)
        for cx, cy in ((r, r), (w_m - r, r), (r, h_m - r), (w_m - r, h_m - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
        m3 = (mask[..., None] / 255.0) * shadow_alpha
        img[sy1:sy2, sx1:sx2] = (dark * m3 + roi * (1 - m3)).astype(np.uint8)
    # 카드
    rounded_rect(img, p1, p2, L_CARD, -1, radius)


def draw_header(img, title: str, height: int = 64) -> None:
    """상단 헤더 — 흰색 배경 + 뒤로가기 + 제목 + 우측 아이콘 (in-place)."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, height), L_HEADER_BG, -1)
    cv2.line(img, (0, height), (w, height), L_DIVIDER, 1)
    # 뒤로가기 (왼쪽 화살표)
    cv2.line(img, (24, height // 2), (40, height // 2 - 8), L_TEXT, 2, cv2.LINE_AA)
    cv2.line(img, (24, height // 2), (40, height // 2 + 8), L_TEXT, 2, cv2.LINE_AA)
    # 제목
    img[:] = kr(img, title, (60, 18), size=22, color=L_TEXT)
    # 우측 아이콘들 (차트 + 설정)
    cx_chart = w - 70
    for i, (x, y, lab) in enumerate([(0, 0, "📊"), (40, 0, "⚙️")]):
        cx = w - 70 + i * 40
        # 차트 (간단한 막대 3개)
        if i == 0:
            for j, h_bar in enumerate((10, 18, 14)):
                cv2.rectangle(img, (cx + j * 6, height // 2 - h_bar // 2),
                              (cx + j * 6 + 4, height // 2 + h_bar // 2), L_TEXT, -1)
        else:
            # 설정 아이콘 (간단한 톱니바퀴 — 원 + 점)
            cv2.circle(img, (cx + 8, height // 2), 9, L_TEXT, 2, cv2.LINE_AA)
            cv2.circle(img, (cx + 8, height // 2), 3, L_TEXT, -1, cv2.LINE_AA)


def draw_workout_info_bar(img, label: str, time_str: str, y: int = 80, height: int = 52) -> None:
    """헤더 아래 운동 정보 박스 — '하체 운동: 스쿼트  05:18' 같은 줄"""
    h, w = img.shape[:2]
    x1, x2 = 16, w - 16
    soft_shadow_card(img, (x1, y), (x2, y + height), radius=14, shadow_offset=3, shadow_alpha=0.08)
    # 아이콘 (덤벨 같은 모양 - 단순 사각형)
    cv2.rectangle(img, (x1 + 18, y + 22), (x1 + 22, y + 30), L_TEXT, -1)
    cv2.rectangle(img, (x1 + 22, y + 18), (x1 + 38, y + 34), L_TEXT, -1)
    cv2.rectangle(img, (x1 + 38, y + 22), (x1 + 42, y + 30), L_TEXT, -1)
    img[:] = kr(img, label, (x1 + 58, y + 14), size=18, color=L_TEXT)
    tw = kr_w(time_str, 18)
    img[:] = kr(img, time_str, (x2 - tw - 18, y + 14), size=18, color=L_SUB)


def draw_stat_card(img, x: int, y: int, w: int, label: str, value: str,
                   value_color=L_TEXT, sub: str = "") -> None:
    """오른쪽 정보 카드 — 정확도/반복/세트/칼로리"""
    h = 78 if sub else 66
    soft_shadow_card(img, (x, y), (x + w, y + h), radius=12, shadow_offset=3, shadow_alpha=0.08)
    img[:] = kr(img, label, (x + 14, y + 10), size=13, color=L_SUB)
    img[:] = kr(img, value, (x + 14, y + 28), size=28, color=value_color)
    if sub:
        img[:] = kr(img, sub, (x + 14, y + 56), size=12, color=value_color)


def draw_feedback_row(img, x: int, y: int, w: int, kind: str, text: str) -> int:
    """실시간 피드백 카드 한 줄 — 체크/경고/정보 아이콘 + 메시지.

    kind: 'check' (녹색 ✓), 'warn' (주황 ⚠), 'info' (파랑 i)
    return: 다음 줄 y
    """
    if kind == "check":
        icon_color = L_GREEN_DARK; icon_bg = L_GREEN_BG; sym = "✓"
    elif kind == "warn":
        icon_color = L_AMBER;       icon_bg = L_AMBER_BG; sym = "!"
    else:
        icon_color = L_BLUE;        icon_bg = L_BLUE_BG;  sym = "i"
    line_h = 36
    # 아이콘 (둥근 박스)
    rounded_rect(img, (x, y + 4), (x + 26, y + 30), icon_bg, -1, 8)
    sw = kr_w(sym, 16)
    img[:] = kr(img, sym, (x + (26 - sw) // 2, y + 7), size=16, color=icon_color)
    # 텍스트
    img[:] = kr(img, text, (x + 36, y + 8), size=15, color=L_TEXT)
    return y + line_h


def draw_score_pill(img, x: int, y: int, w: int, h: int, accuracy: int,
                    label: str = "자세 정확도", remark: str = "좋아요!") -> None:
    """우측 패널 — 큰 점수 표시 (이미지의 87% 좋아요! 같은)"""
    soft_shadow_card(img, (x, y), (x + w, y + h), radius=14, shadow_offset=3, shadow_alpha=0.08)
    img[:] = kr(img, label, (x + 14, y + 12), size=13, color=L_SUB)
    big = f"{accuracy}%"
    img[:] = kr(img, big, (x + 14, y + 30), size=42, color=L_GREEN_DARK)
    img[:] = kr(img, remark, (x + 14, y + 84), size=14, color=L_GREEN_DARK)


def draw_metric_row(img, x: int, y: int, w: int, h: int, label: str, value: str,
                    sub_label: str = "") -> None:
    soft_shadow_card(img, (x, y), (x + w, y + h), radius=12, shadow_offset=3, shadow_alpha=0.08)
    img[:] = kr(img, label, (x + 14, y + 10), size=13, color=L_SUB)
    img[:] = kr(img, value, (x + 14, y + 28), size=22, color=L_TEXT)
    if sub_label:
        img[:] = kr(img, sub_label, (x + 14, y + 56), size=12, color=L_SUB)


def draw_camera_frame(img, x: int, y: int, w: int, h: int, cam_bgr: Optional[np.ndarray] = None,
                      label: str = "LIVE  On-device AI",
                      landmarks=None, pose_key: Optional[str] = None,
                      highlight_indices=None) -> None:
    """카메라 영역 — 둥근 모서리 + 좌하단 LIVE 배지 + 골격 오버레이 옵션."""
    soft_shadow_card(img, (x, y), (x + w, y + h), radius=18, shadow_offset=5, shadow_alpha=0.10)
    cam_w = w - 8; cam_h = h - 8
    cx, cy = x + 4, y + 4
    if cam_bgr is not None:
        cam = cv2.resize(cam_bgr, (cam_w, cam_h))
        # 골격 오버레이 (resize된 cam 위에 직접 그림)
        if landmarks is not None:
            _draw_skeleton_on(cam, landmarks, pose_key=pose_key, highlight=highlight_indices)
        img[cy:cy + cam_h, cx:cx + cam_w] = cam
    else:
        cv2.rectangle(img, (cx, cy), (cx + cam_w, cy + cam_h), (235, 235, 240), -1)
    # LIVE 배지
    badge_w = kr_w(label, 13) + 24
    bx, by = x + 16, y + h - 36
    rounded_rect(img, (bx, by), (bx + badge_w, by + 26), (45, 45, 45), -1, 13)
    cv2.circle(img, (bx + 12, by + 13), 4, L_GREEN, -1)
    img[:] = kr(img, label, (bx + 22, by + 4), size=13, color=(245, 245, 245))


def _draw_skeleton_on(cam_bgr, landmarks, pose_key=None, highlight=None,
                       vis_threshold: float = 0.5) -> None:
    """카메라 프레임 위에 직접 골격 그리기 (in-place).

    vis_threshold 0.5 → 정말 잘 보이는 관절만 그림.
    낮은 visibility의 추정값은 화면에 표시하지 않음 (혼란 방지).
    """
    if landmarks is None:
        return
    h, w = cam_bgr.shape[:2]
    BONES = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    KEY = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    bone_color = (250, 200, 80)
    point_color = (80, 220, 80)
    hl_color = (60, 220, 60)
    pts = {}
    for idx in KEY:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < vis_threshold:
            continue
        # 좌표가 frame 범위 밖이면 거부 (lifter가 out-of-frame 추정한 outlier)
        x, y = float(lm.x), float(lm.y)
        if not (-0.05 < x < 1.05 and -0.05 < y < 1.05):
            continue
        pts[idx] = (int(x * w), int(y * h))
    # 본
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(cam_bgr, pts[a], pts[b], (255, 255, 255), 6, cv2.LINE_AA)
            cv2.line(cam_bgr, pts[a], pts[b], bone_color, 3, cv2.LINE_AA)
    # 점
    for idx, p in pts.items():
        if highlight and idx in highlight:
            cv2.circle(cam_bgr, p, 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(cam_bgr, p, 7, hl_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(cam_bgr, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(cam_bgr, p, 4, point_color, -1, cv2.LINE_AA)


def draw_pose_mini(img, x: int, y: int, w: int, h: int, pose_key: str,
                   angle_a_label: str, angle_a_value: str,
                   angle_b_label: str, angle_b_value: str) -> None:
    """미니 자세 일러스트 + 각도 표시 (참고 이미지의 흰 인형)"""
    soft_shadow_card(img, (x, y), (x + w, y + h), radius=14, shadow_offset=3, shadow_alpha=0.08)
    # 자세 일러스트
    from .minimal_ui import _draw_pose_icon_minimal
    _draw_pose_icon_minimal(img, x + 16, y + 12, x + w - 80, y + h - 12, pose_key,
                            color=(180, 180, 180))
    # 우측에 각도 두 개
    img[:] = kr(img, angle_a_label, (x + w - 76, y + 14), size=12, color=L_SUB)
    img[:] = kr(img, angle_a_value, (x + w - 76, y + 30), size=20, color=L_GREEN_DARK)
    img[:] = kr(img, angle_b_label, (x + w - 76, y + 60), size=12, color=L_SUB)
    img[:] = kr(img, angle_b_value, (x + w - 76, y + 76), size=20, color=L_GREEN_DARK)


def draw_mode_strip(img, y: int, height: int = 44, items=None) -> None:
    """하단 모드 표시 — '온디바이스', '오프라인', '개인정보보호' 등"""
    h, w = img.shape[:2]
    if items is None:
        items = [("On-device", "기기 내 처리"),
                 ("오프라인 모드", "인터넷 불필요"),
                 ("개인정보 보호", "영상 외부 전송 없음")]
    cv2.rectangle(img, (0, y), (w, y + height), L_GREEN_BG, -1)
    n = len(items)
    cell = w // n
    for i, (title, sub) in enumerate(items):
        cx = i * cell + 18
        # 아이콘 (간단한 점)
        cv2.circle(img, (cx + 6, y + 17), 5, L_GREEN_DARK, -1)
        img[:] = kr(img, title, (cx + 18, y + 8), size=13, color=L_TEXT)
        img[:] = kr(img, sub, (cx + 18, y + 24), size=11, color=L_SUB)


def draw_action_button(img, x: int, y: int, w: int, h: int, label: str,
                       primary: bool = False) -> None:
    color = L_TEXT if primary else (235, 235, 240)
    text_color = (240, 240, 240) if primary else L_TEXT
    rounded_rect(img, (x, y), (x + w, y + h), color, -1, h // 2)
    # 아이콘 (primary는 ▶, secondary는 ■)
    if primary:
        # ▶ 일시정지 아이콘 (두 막대)
        bar_w = 4; bar_h = 16
        cx = x + w // 2 - 28
        cy = y + h // 2
        cv2.rectangle(img, (cx, cy - bar_h // 2), (cx + bar_w, cy + bar_h // 2), text_color, -1)
        cv2.rectangle(img, (cx + bar_w + 4, cy - bar_h // 2),
                      (cx + bar_w + 4 + bar_w, cy + bar_h // 2), text_color, -1)
    else:
        cx = x + w // 2 - 28; cy = y + h // 2
        cv2.rectangle(img, (cx, cy - 7), (cx + 14, cy + 7), L_RED, -1)
    tw = kr_w(label, 16)
    img[:] = kr(img, label, (x + (w - tw) // 2 + 6, y + (h - 16) // 2),
                size=16, color=text_color)


def fill_bg(img) -> None:
    img[:] = L_BG
