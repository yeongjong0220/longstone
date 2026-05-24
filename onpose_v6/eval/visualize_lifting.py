"""
GT 3D vs Pred 3D 골격을 나란히 비교하는 시각화 영상 생성.

발표/시연용 — 학습된 lifter가 얼마나 잘 GT를 따라가는지 한눈에 보여줌.
파란 = GT, 빨강 = Pred, 두 골격을 같은 캔버스에 겹쳐서 그림.

사용:
  python eval/visualize_lifting.py \
    --pred-csv ../pilates_temporal_lifter/predicted_eval_progress3_angle_causal_v1.csv \
    --gt-csv  ../pilates_temporal_lifter/the_seal_gt3d_trim.csv \
    --out reports/compare_the_seal.mp4 --fps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from core.lifting_postprocess import JOINT_NAMES, JOINT_IDX, df_to_seq, enforce_bone_length
from core.ui_helpers import (COLOR_BAD, COLOR_DIM, COLOR_OK, COLOR_PANEL_BORDER, COLOR_TEXT,
                             COLOR_WARN, draw_kr, measure_kr)

BONES = [
    ("Head", "Neck"), ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("Neck", "Hip"),
    ("Hip", "LHip"), ("Hip", "RHip"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "Rknee"), ("Rknee", "RAnkle"),
]


def normalize_skeleton_3d(seq):
    """Hip-root 중심, 토르소 길이 1.0 단위로 정규화"""
    h = JOINT_IDX["Hip"]
    n = JOINT_IDX["Neck"]
    seq = seq - seq[:, [h], :]
    torso = np.linalg.norm(seq[:, n] - seq[:, h], axis=-1).mean()
    return seq / max(torso, 1e-8)


def project_xy(seq3d, canvas_w, canvas_h, cx_offset=0, scale=140.0):
    """간단한 정사영(XY) — Z는 무시. 사람이 정면을 보는 frame이라 충분."""
    cx = canvas_w // 2 + cx_offset
    cy = canvas_h // 2 + 50
    # x 그대로, y는 위가 +Y (다양한 데이터셋 호환을 위해 부호 반전 옵션 검사)
    proj = seq3d[..., :2].copy()
    # heuristic: 머리가 발보다 작은 y 값을 가져야 자연스럽게 보이도록
    h_idx = JOINT_IDX["Head"]
    a_idx = JOINT_IDX["LAnkle"]
    mean_head_y = proj[:, h_idx, 1].mean()
    mean_ankle_y = proj[:, a_idx, 1].mean()
    if mean_head_y > mean_ankle_y:
        proj[..., 1] = -proj[..., 1]
    px = (proj[..., 0] * scale + cx).astype(int)
    py = (proj[..., 1] * scale + cy).astype(int)
    return np.stack([px, py], axis=-1)


def draw_skeleton(img, pts2d, color, thickness=2, point_r=4):
    for a, b in BONES:
        if a not in JOINT_IDX or b not in JOINT_IDX:
            continue
        pa = tuple(pts2d[JOINT_IDX[a]])
        pb = tuple(pts2d[JOINT_IDX[b]])
        cv2.line(img, pa, pb, color, thickness)
    for j, p in enumerate(pts2d):
        cv2.circle(img, tuple(p), point_r, color, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--apply-postproc", action="store_true",
                    help="Pred에 bone-length lock 적용 후 비교")
    ap.add_argument("--smooth-gt", action="store_true",
                    help="GT 키포인트에 Savitzky-Golay 스무딩 적용 (GT 자체 jitter 완화)")
    ap.add_argument("--smooth-pred", action="store_true",
                    help="Pred에 동일 스무딩 적용 (공정 비교용)")
    ap.add_argument("--smooth-window", type=int, default=11)
    ap.add_argument("--smooth-poly", type=int, default=3)
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    gt_df = pd.read_csv(args.gt_csv)
    n = min(len(pred_df), len(gt_df))
    pred_seq = df_to_seq(pred_df.iloc[:n])
    gt_seq = df_to_seq(gt_df.iloc[:n])

    # GT 자체 스무딩 (raw GT가 jittery한 경우)
    if args.smooth_gt:
        try:
            from scipy.signal import savgol_filter
            w = args.smooth_window if args.smooth_window % 2 == 1 else args.smooth_window + 1
            for j in range(gt_seq.shape[1]):
                for c in range(gt_seq.shape[2]):
                    gt_seq[:, j, c] = savgol_filter(gt_seq[:, j, c], w, args.smooth_poly, mode="interp")
            print(f"[smooth] GT Savitzky-Golay window={w} poly={args.smooth_poly}")
        except ImportError:
            print("[warn] scipy not available, skipping GT smoothing")
    if args.smooth_pred:
        try:
            from scipy.signal import savgol_filter
            w = args.smooth_window if args.smooth_window % 2 == 1 else args.smooth_window + 1
            for j in range(pred_seq.shape[1]):
                for c in range(pred_seq.shape[2]):
                    pred_seq[:, j, c] = savgol_filter(pred_seq[:, j, c], w, args.smooth_poly, mode="interp")
            print(f"[smooth] Pred Savitzky-Golay window={w} poly={args.smooth_poly}")
        except ImportError:
            pass

    if args.apply_postproc:
        pred_seq = enforce_bone_length(pred_seq)
    pred_seq = normalize_skeleton_3d(pred_seq)
    gt_seq = normalize_skeleton_3d(gt_seq)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (args.width, args.height))
    print(f"[render] {n} frames -> {args.out}")

    for t in range(n):
        canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
        # Left half = GT
        gt_pts = project_xy(gt_seq[t:t+1], args.width // 2, args.height,
                            cx_offset=0, scale=180)[0]
        draw_skeleton(canvas[:, :args.width // 2], gt_pts, (240, 200, 80), thickness=3, point_r=5)
        # Right half = Pred
        pred_pts = project_xy(pred_seq[t:t+1], args.width // 2, args.height,
                              cx_offset=0, scale=180)[0]
        right = canvas[:, args.width // 2:].copy()
        draw_skeleton(right, pred_pts, (90, 110, 240), thickness=3, point_r=5)
        canvas[:, args.width // 2:] = right
        # Center divider
        cv2.line(canvas, (args.width // 2, 60), (args.width // 2, args.height - 30),
                 (60, 60, 80), 1)
        # 헤더
        canvas = draw_kr(canvas, "GT (정답 3D)", (40, 24), size=28, color=(80, 200, 240))
        canvas = draw_kr(canvas, "Pred (lifter 결과)", (args.width // 2 + 40, 24),
                         size=28, color=(240, 110, 90))
        canvas = draw_kr(canvas, f"frame {t+1}/{n}",
                         (args.width - 220, 24), size=22, color=COLOR_DIM)
        # Per-frame error
        err = float(np.linalg.norm(pred_seq[t] - gt_seq[t], axis=-1).mean())
        col = COLOR_OK if err < 0.3 else (COLOR_WARN if err < 0.5 else COLOR_BAD)
        canvas = draw_kr(canvas, f"MPJPE (root-norm) = {err:.3f}",
                         (40, args.height - 26), size=18, color=col)
        writer.write(canvas)

    writer.release()
    print(f"[done] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
