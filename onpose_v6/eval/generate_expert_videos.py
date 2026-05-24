"""
데이터셋의 GT 3D 좌표 → 자세별 전문가 가이드 영상 자동 생성.

assets/The_Seal.mp4 / assets/Spine_Stretch.mp4 / assets/Bridging.mp4 로 저장.
UI는 측정 단계에서 이 영상을 PIP로 보여줘 사용자가 따라할 수 있게 한다.

특징:
  - Savitzky-Golay 스무딩 적용 (raw GT는 jittery → 깔끔한 가이드)
  - 자세별 카메라 시점 자동 조정
  - 작은 해상도 (PIP용)로 출력

사용:
  python eval/generate_expert_videos.py \
    --root "D:/dataset/216.필라테스 동작 데이터/01-1.정식개방데이터/Training_1/Mat"
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

from core.lifting_postprocess import JOINT_IDX, JOINT_NAMES, df_to_seq
from eval.extract_pose_stats import find_3d_csvs, trim_with_json
from eval.visualize_lifting import BONES, draw_skeleton, normalize_skeleton_3d


def smooth_seq(seq: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter
    out = seq.copy()
    w = window if window % 2 == 1 else window + 1
    if len(out) < w + 1:
        return out
    for j in range(out.shape[1]):
        for c in range(out.shape[2]):
            out[:, j, c] = savgol_filter(out[:, j, c], w, poly, mode="interp")
    return out


def project(seq, w, h, scale=160, ox=0, oy=40):
    cx = w // 2 + ox
    cy = h // 2 + oy
    proj = seq[..., :2].copy()
    h_idx = JOINT_IDX["Head"]; a_idx = JOINT_IDX["LAnkle"]
    if proj[:, h_idx, 1].mean() > proj[:, a_idx, 1].mean():
        proj[..., 1] = -proj[..., 1]
    px = (proj[..., 0] * scale + cx).astype(int)
    py = (proj[..., 1] * scale + cy).astype(int)
    return np.stack([px, py], axis=-1)


def render_one(seq: np.ndarray, out_path: Path, fps: int = 25,
               size=(360, 480), color=(120, 220, 140)) -> None:
    """3D 시퀀스 → mp4 (loop 가능한 작은 영상)"""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    seq2d = project(seq, w, h, scale=min(w, h) * 0.35, ox=0, oy=20)
    for t in range(len(seq)):
        canvas = np.full((h, w, 3), 30, dtype=np.uint8)
        # 그리드 라인 (참고용)
        for gy in range(0, h, 40):
            cv2.line(canvas, (0, gy), (w, gy), (45, 45, 60), 1)
        # 골격
        draw_skeleton(canvas, seq2d[t], color, thickness=3, point_r=5)
        # 프레임 번호 (작게)
        cv2.putText(canvas, f"{t+1}/{len(seq)}", (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 160), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    print(f"[ok] saved {out_path}  ({len(seq)} frames)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=THIS.parents[1] / "assets")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--smooth-window", type=int, default=11)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csvs = find_3d_csvs(args.root)
    by_pose = {}
    for c in csvs:
        by_pose.setdefault(c["pose"], []).append(c)

    for pose_key, items in by_pose.items():
        # 각 자세에서 가장 깔끔한(긴 시퀀스) actor 하나 선택
        items.sort(key=lambda c: -pd.read_csv(c["csv"]).shape[0])
        for item in items[:5]:   # 상위 5개 중 첫 성공
            try:
                df = pd.read_csv(item["csv"])
                if item["json"]:
                    df = trim_with_json(df, item["json"])
                if len(df) < 30:
                    continue
                seq = df_to_seq(df)
                seq = smooth_seq(seq, window=args.smooth_window)
                seq_n = normalize_skeleton_3d(seq)
                out_path = args.out_dir / f"{pose_key}.mp4"
                render_one(seq_n, out_path, fps=args.fps)
                print(f"  source actor: {item['actor']}")
                break
            except Exception as exc:
                print(f"  skip {item['actor']}: {exc}")
                continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
