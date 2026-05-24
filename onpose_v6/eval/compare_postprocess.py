"""
Lifting 결과 + 후처리 효과 비교 평가 스크립트.

사용 예:
  python eval/compare_postprocess.py \
    --pred-csv ../pilates_temporal_lifter/predicted_eval_progress3_angle_causal_v1.csv \
    --gt-csv  ../pilates_temporal_lifter/the_seal_gt3d_trim.csv

PowerShell 사용 시 백틱(`)으로 줄바꿈하거나 한 줄로 입력하세요.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from core.lifting_postprocess import (
    JOINT_NAMES,
    JOINT_IDX,
    df_to_seq,
    seq_to_df,
    enforce_bone_length,
    temporal_smooth_3d,
    clip_velocity,
    postprocess_pipeline,
    ensemble_average,
)
from eval.lifting_accuracy import evaluate


def evaluate_seq(pred_seq: np.ndarray, base_df: pd.DataFrame, gt_csv: Path, pose_key: str = "the_seal"):
    """후처리된 seq를 임시 CSV로 만들어 evaluate() 호출"""
    tmp_df = seq_to_df(pred_seq, base_df=base_df)
    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        tmp_df.to_csv(tmp_path, index=False)
        return evaluate(Path(tmp_path), gt_csv, pose_key=pose_key)
    finally:
        os.unlink(tmp_path)


def headline(s: dict) -> dict:
    return {
        "MAE_hip": s["angle_mae_deg"]["hip"],
        "MAE_knee": s["angle_mae_deg"]["knee"],
        "MAE_trunk": s["angle_mae_deg"]["trunk"],
        "PCK_hip_15": s["angle_pck"]["@15deg"]["hip"],
        "PCK_knee_15": s["angle_pck"]["@15deg"]["knee"],
        "PCK_hip_20": s["angle_pck"]["@20deg"]["hip"],
        "PCK_knee_20": s["angle_pck"]["@20deg"]["knee"],
        "OnPose_core": s["onpose_accuracy_pct_core_hip_knee"],
        "MPJPE_rn": s["mpjpe_root_norm"],
    }


def print_table(rows: list[tuple[str, dict]]):
    cols = ["MAE_hip", "MAE_knee", "MAE_trunk", "PCK_hip_15", "PCK_knee_15",
            "PCK_hip_20", "PCK_knee_20", "OnPose_core", "MPJPE_rn"]
    head = f"{'method':<28}" + "".join(f"{c:>14}" for c in cols)
    print(head)
    print("-" * len(head))
    for name, d in rows:
        row = f"{name:<28}"
        for c in cols:
            v = d[c]
            if "MAE" in c:
                row += f"{v:>13.2f}d"
            elif "MPJPE" in c:
                row += f"{v:>14.4f}"
            else:
                row += f"{v:>13.2f}%"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare lifting accuracy with/without postprocessing")
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--extra-pred-csv", type=Path, default=None,
                    help="Optional second model output (for ensemble test)")
    ap.add_argument("--pose", type=str, default="the_seal")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    pred_seq = df_to_seq(pred_df)

    # Baseline
    baseline = evaluate(args.pred_csv, args.gt_csv, pose_key=args.pose)

    # 1) clip velocity (outlier spike 제거)
    s_clip = clip_velocity(pred_seq, max_step=0.10)
    r_clip = evaluate_seq(s_clip, pred_df, args.gt_csv, args.pose)

    # 2) bone-length enforce
    s_bone = enforce_bone_length(pred_seq)
    r_bone = evaluate_seq(s_bone, pred_df, args.gt_csv, args.pose)

    # 3) temporal smoothing (Savitzky-Golay)
    s_smooth = temporal_smooth_3d(pred_seq, method="savgol", window=11, polyorder=3)
    r_smooth = evaluate_seq(s_smooth, pred_df, args.gt_csv, args.pose)

    # 4) Combined recommended pipeline
    s_full = postprocess_pipeline(pred_seq, smooth_window=11, do_bone_lock=True,
                                  do_velocity_clip=True, max_velocity_step=0.10)
    r_full = evaluate_seq(s_full, pred_df, args.gt_csv, args.pose)

    # 5) Smoothing only (window 변경 비교용)
    s_smooth_short = temporal_smooth_3d(pred_seq, method="savgol", window=7, polyorder=3)
    r_smooth_short = evaluate_seq(s_smooth_short, pred_df, args.gt_csv, args.pose)

    # 6) Ensemble with extra model (옵션)
    rows = [
        ("baseline (no postproc)", headline(baseline)),
        ("[1] velocity clip 0.10", headline(r_clip)),
        ("[2] bone-length enforce", headline(r_bone)),
        ("[3] smooth savgol w=7", headline(r_smooth_short)),
        ("[3] smooth savgol w=11", headline(r_smooth)),
        ("[ALL] vel+bone+smooth", headline(r_full)),
    ]

    if args.extra_pred_csv is not None and args.extra_pred_csv.exists():
        extra_df = pd.read_csv(args.extra_pred_csv)
        extra_seq = df_to_seq(extra_df)
        # 시퀀스 길이 맞춤
        n = min(len(pred_seq), len(extra_seq))
        ens = ensemble_average([pred_seq[:n], extra_seq[:n]], weights=[0.5, 0.5])
        r_ens = evaluate_seq(ens, pred_df.iloc[:n], args.gt_csv, args.pose)
        s_ens_full = postprocess_pipeline(ens, smooth_window=11)
        r_ens_full = evaluate_seq(s_ens_full, pred_df.iloc[:n], args.gt_csv, args.pose)
        rows.append((f"[ENS] avg w/ extra model", headline(r_ens)))
        rows.append((f"[ENS+ALL] postproc on ens", headline(r_ens_full)))

    print()
    print_table(rows)

    # 가장 좋은 조합 표시
    best_pck = max(rows, key=lambda r: r[1]["PCK_hip_15"])
    print()
    print(f"==> Best PCK_hip_15: {best_pck[0]}  ({best_pck[1]['PCK_hip_15']:.2f}%)")
    best_onpose = max(rows, key=lambda r: r[1]["OnPose_core"])
    print(f"==> Best OnPose_core: {best_onpose[0]}  ({best_onpose[1]['OnPose_core']:.2f}%)")

    if args.out:
        out = {name: data for name, data in rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
