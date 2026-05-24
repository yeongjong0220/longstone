"""
OpenCV 기반 경량 차트 (matplotlib 의존 없음 → 모바일/임베디드 친화).

- sparkline_score(scores) : 5초 캡처 동안 점수 변화 라인 그래프
- bars_per_angle(...)     : 각도별 정확도 막대
- radial_score(score)     : 원형 점수 다이얼 (게이지)
- history_chart(scores)   : 세션 히스토리 도트
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .ui_helpers import (COLOR_BAD, COLOR_DIM, COLOR_OK, COLOR_PANEL, COLOR_PANEL_BORDER,
                         COLOR_TEXT, COLOR_WARN, draw_kr, measure_kr, make_panel_image)


def _pick_color(score: float) -> Tuple[int, int, int]:
    if score >= 85:
        return COLOR_OK
    if score >= 70:
        return COLOR_WARN
    return COLOR_BAD


def sparkline_score(scores: List[float], size: Tuple[int, int] = (460, 110),
                    title: str = "프레임별 점수") -> np.ndarray:
    """캡처 동안의 점수 시계열 라인"""
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=16, color=COLOR_DIM)
    if not scores:
        return img
    arr = np.asarray(scores, dtype=np.float32)
    mn, mx = max(0, arr.min() - 5), min(100, arr.max() + 5)
    if mx - mn < 10:
        mn, mx = max(0, arr.mean() - 10), min(100, arr.mean() + 10)
    # 격자
    top, bot = 30, h - 18
    cv2.line(img, (10, top), (w - 10, top), (70, 70, 90), 1)
    cv2.line(img, (10, bot), (w - 10, bot), (70, 70, 90), 1)
    cv2.line(img, (10, (top + bot) // 2), (w - 10, (top + bot) // 2), (60, 60, 75), 1)
    # 라인
    pts = []
    for i, s in enumerate(arr):
        x = 10 + int((w - 20) * (i / max(1, len(arr) - 1)))
        y = int(bot - (bot - top) * ((s - mn) / max(1e-6, mx - mn)))
        pts.append((x, y))
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            c = _pick_color(float(arr[i + 1]))
            cv2.line(img, pts[i], pts[i + 1], c, 2)
    # 마지막 점
    if pts:
        cv2.circle(img, pts[-1], 4, COLOR_TEXT, -1)
    # 축 라벨
    img = draw_kr(img, f"{int(mx)}", (w - 30, top - 2), size=12, color=COLOR_DIM)
    img = draw_kr(img, f"{int(mn)}", (w - 30, bot - 14), size=12, color=COLOR_DIM)
    avg = float(arr.mean())
    img = draw_kr(img, f"평균 {avg:.0f}", (w - 110, 6), size=14,
                  color=_pick_color(avg))
    return img


def bars_per_angle(per_angle_accuracy: Dict[str, float], per_angle_diff: Dict[str, float],
                   labels_kr: Dict[str, str], size: Tuple[int, int] = (460, 150),
                   title: str = "각도별 정확도 (가중치 적용)") -> np.ndarray:
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=16, color=COLOR_DIM)
    rows = list(per_angle_accuracy.items())
    if not rows:
        return img
    row_h = (h - 36) // max(1, len(rows))
    for i, (key, pct) in enumerate(rows):
        yy = 32 + i * row_h
        label = labels_kr.get(key, key)
        img = draw_kr(img, label, (10, yy), size=14, color=COLOR_TEXT)
        bar_x = 110
        bar_w = w - bar_x - 110
        cv2.rectangle(img, (bar_x, yy + 4), (bar_x + bar_w, yy + 4 + row_h - 12), (55, 55, 70), -1)
        fill = int(bar_w * max(0.0, min(1.0, pct / 100.0)))
        color = _pick_color(pct)
        cv2.rectangle(img, (bar_x, yy + 4), (bar_x + fill, yy + 4 + row_h - 12), color, -1)
        diff = per_angle_diff.get(key, 0.0)
        img = draw_kr(img, f"{pct:.0f}%  (Δ{diff:.1f}°)", (bar_x + bar_w + 6, yy), size=14,
                      color=COLOR_TEXT)
    return img


def radial_score(score: float, size: Tuple[int, int] = (150, 150),
                 label: str = "점수") -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 38, dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 8
    # 외곽 어두운 링
    cv2.circle(img, (cx, cy), r, (70, 70, 90), 5)
    # 점수에 따른 호
    angle = int(360 * max(0.0, min(100.0, score)) / 100.0)
    color = _pick_color(score)
    if angle > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, angle, color, 8)
    img = draw_kr(img, f"{score:.0f}", (cx - 28, cy - 24), size=44, color=COLOR_TEXT)
    img = draw_kr(img, label, (cx - measure_kr(label, 16) // 2, cy + 24), size=16, color=COLOR_DIM)
    return img


def history_chart(history: List[Tuple[str, float]], size: Tuple[int, int] = (460, 90),
                  title: str = "세션 히스토리 (최근 5회)") -> np.ndarray:
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=16, color=COLOR_DIM)
    if not history:
        img = draw_kr(img, "아직 기록이 없어요", (10, 36), size=14, color=COLOR_DIM)
        return img
    n = len(history)
    spacing = (w - 30) / max(1, n)
    for i, (pose_kr, sc) in enumerate(history):
        x = int(15 + spacing * (i + 0.5))
        y = h - 25 - int((h - 50) * (sc / 100.0))
        cv2.circle(img, (x, y), 6, _pick_color(sc), -1)
        img = draw_kr(img, f"{sc:.0f}", (x - 10, y - 22), size=14, color=COLOR_TEXT)
        img = draw_kr(img, pose_kr[:6], (x - 18, h - 18), size=11, color=COLOR_DIM)
        if i + 1 < n:
            x2 = int(15 + spacing * (i + 1.5))
            y2 = h - 25 - int((h - 50) * (history[i + 1][1] / 100.0))
            cv2.line(img, (x, y), (x2, y2), (90, 90, 110), 1)
    return img


def angle_timeline(angle_series: Dict[str, List[float]], targets: Dict[str, float],
                   tolerances: Dict[str, float], size: Tuple[int, int] = (460, 160),
                   title: str = "각도 시계열 (정답 라인 = 점선)") -> np.ndarray:
    """캡처 5초 동안의 hip / knee / trunk 각도 변화 + 정답 라인 + tolerance 밴드"""
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=14, color=COLOR_DIM)
    if not angle_series:
        return img
    # 색상 매핑
    palette = {"hip": (240, 180, 80), "knee": (80, 220, 140), "trunk": (90, 110, 240)}
    label_kr = {"hip": "고관절", "knee": "무릎", "trunk": "상체"}
    # 전체 y 범위
    all_vals = []
    for v in angle_series.values():
        all_vals.extend(v)
    for v in targets.values():
        all_vals.append(v)
    if not all_vals:
        return img
    mn, mx = min(all_vals) - 5, max(all_vals) + 5
    top, bot = 28, h - 16
    # 격자
    cv2.line(img, (10, top), (w - 10, top), (70, 70, 90), 1)
    cv2.line(img, (10, bot), (w - 10, bot), (70, 70, 90), 1)
    # 각 각도 라인
    for key, series in angle_series.items():
        if not series:
            continue
        color = palette.get(key, (200, 200, 200))
        # 정답 라인 (점선)
        if key in targets:
            ty = int(bot - (bot - top) * ((targets[key] - mn) / max(1e-6, mx - mn)))
            for x in range(10, w - 10, 6):
                cv2.line(img, (x, ty), (x + 3, ty), color, 1)
            # tolerance 밴드 음영
            if key in tolerances:
                tol = tolerances[key]
                y_hi = int(bot - (bot - top) * ((targets[key] - tol - mn) / max(1e-6, mx - mn)))
                y_lo = int(bot - (bot - top) * ((targets[key] + tol - mn) / max(1e-6, mx - mn)))
                overlay = img.copy()
                cv2.rectangle(overlay, (10, min(y_hi, y_lo)), (w - 10, max(y_hi, y_lo)),
                              color, -1)
                img = cv2.addWeighted(overlay, 0.10, img, 0.90, 0)
        # 실제 측정 선
        arr = np.asarray(series, dtype=np.float32)
        pts = []
        for i, val in enumerate(arr):
            x = 10 + int((w - 20) * (i / max(1, len(arr) - 1)))
            y = int(bot - (bot - top) * ((val - mn) / max(1e-6, mx - mn)))
            pts.append((x, y))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 2)
    # 범례
    lx = w - 230
    ly = 6
    for key, color in palette.items():
        if key in angle_series:
            cv2.rectangle(img, (lx, ly + 2), (lx + 14, ly + 14), color, -1)
            img = draw_kr(img, label_kr.get(key, key), (lx + 18, ly), size=12, color=COLOR_TEXT)
            lx += 70
    return img


def pose_trend(pose_history: Dict[str, List[float]], size: Tuple[int, int] = (460, 140),
               title: str = "자세별 점수 추세") -> np.ndarray:
    """자세별 점수 흐름을 라인으로 — 사용자가 늘고 있는지 한눈에 확인"""
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=14, color=COLOR_DIM)
    if not pose_history or not any(pose_history.values()):
        img = draw_kr(img, "아직 자세별 기록이 없어요", (10, 36), size=14, color=COLOR_DIM)
        return img
    palette = {"더 씰": (240, 180, 80), "브릿징": (80, 220, 140), "스파인 스트레치": (90, 110, 240)}
    top, bot = 30, h - 20
    cv2.line(img, (10, top), (w - 10, top), (70, 70, 90), 1)
    cv2.line(img, (10, bot), (w - 10, bot), (70, 70, 90), 1)
    # 50 / 100 라벨
    mid = (top + bot) // 2
    cv2.line(img, (10, mid), (w - 10, mid), (55, 55, 75), 1)
    img = draw_kr(img, "100", (w - 30, top - 2), size=11, color=COLOR_DIM)
    img = draw_kr(img, "50", (w - 25, mid - 6), size=11, color=COLOR_DIM)
    img = draw_kr(img, "0", (w - 20, bot - 12), size=11, color=COLOR_DIM)

    # 각 자세별 라인
    legend_x = 10
    for pose_kr, scores in pose_history.items():
        if not scores:
            continue
        color = palette.get(pose_kr, (180, 180, 180))
        pts = []
        for i, s in enumerate(scores):
            x = 10 + int((w - 20) * (i / max(1, len(scores) - 1)))
            y = int(bot - (bot - top) * (s / 100.0))
            pts.append((x, y))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 2)
        # 마지막 점
        if pts:
            cv2.circle(img, pts[-1], 4, color, -1)
            avg = float(np.mean(scores))
            img = draw_kr(img, f"{pose_kr}: 평균 {avg:.0f}", (pts[-1][0] + 6, pts[-1][1] - 8),
                          size=11, color=color)
        # 범례
        cv2.rectangle(img, (legend_x, h - 14), (legend_x + 12, h - 4), color, -1)
        img = draw_kr(img, pose_kr, (legend_x + 16, h - 18), size=11, color=COLOR_DIM)
        legend_x += measure_kr(pose_kr, 11) + 50
    return img


def occlusion_bar(visibility_map: Dict[str, float], size: Tuple[int, int] = (460, 80),
                  title: str = "관절 가려짐 상태") -> np.ndarray:
    """주요 관절들의 가려짐(visibility) 상태 표시 — 사용자가 카메라 위치 조정에 도움"""
    w, h = size
    img = make_panel_image(size)
    img = draw_kr(img, title, (10, 6), size=14, color=COLOR_DIM)
    if not visibility_map:
        img = draw_kr(img, "관절 미감지", (10, 36), size=14, color=COLOR_DIM)
        return img
    cols = list(visibility_map.items())
    cell = (w - 20) // max(1, len(cols))
    for i, (name, v) in enumerate(cols):
        x = 10 + i * cell
        color = COLOR_OK if v >= 0.7 else (COLOR_WARN if v >= 0.4 else COLOR_BAD)
        bar_h = int(30 * v)
        cv2.rectangle(img, (x + 4, h - 18 - bar_h), (x + cell - 4, h - 18), color, -1)
        img = draw_kr(img, name, (x + 4, 28), size=12, color=COLOR_DIM)
    return img
