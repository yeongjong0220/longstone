"""
2D→3D Lifting 모델의 정량 평가 — 멘토링 피드백:
"2d keypoint를 3d로 lifting하는 모델 지표가 80% 이상 나오면 좋겠어, 90%면 더 좋고"

평가 지표:
  1) Angle-PCK@τ  : 각도 오차가 τ° 이하인 프레임 비율 (hip / knee / trunk)
     - 가장 직접적인 "자세 정확도"
     - 임계값 10°/15°/20° 별로 계산
  2) Joint-PCK@τ : root-normalized 위치 오차가 τ 이하인 관절-프레임 비율
  3) Per-pose Accuracy : 모든 핵심 각도가 ±15° 이내인 프레임 비율 (OnPose 시스템 자체 정의)
  4) MPJPE (mm 환산은 GT 데이터 단위 따라)

사용:
  python eval/lifting_accuracy.py \
    --pred-csv pilates_temporal_lifter/predicted_eval_progress3_angle_causal_v1.csv \
    --gt-csv  pilates_temporal_lifter/the_seal_gt3d_trim.csv \
    --out reports/lifting_accuracy_the_seal.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


JOINTS = [
    "Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle", "Hip",
]


def vec(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=float)


def angle_deg_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a - b
    v = c - b
    un = np.linalg.norm(u, axis=1, keepdims=True)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    u = u / np.clip(un, 1e-8, None)
    v = v / np.clip(vn, 1e-8, None)
    cos = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_angles(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "hip_left":  angle_deg_batch(vec(df, "Neck"), vec(df, "LHip"), vec(df, "LKnee")),
        "hip_right": angle_deg_batch(vec(df, "Neck"), vec(df, "RHip"), vec(df, "Rknee")),
        "knee_left":  angle_deg_batch(vec(df, "LHip"), vec(df, "LKnee"), vec(df, "LAnkle")),
        "knee_right": angle_deg_batch(vec(df, "RHip"), vec(df, "Rknee"), vec(df, "RAnkle")),
        "trunk":     angle_deg_batch(vec(df, "Hip"),  vec(df, "Neck"),  vec(df, "Head")),
    }


def stack_xyz(df: pd.DataFrame, joints: List[str]) -> np.ndarray:
    return np.stack([df[[f"{j}_x", f"{j}_y", f"{j}_z"]].to_numpy(dtype=float) for j in joints], axis=1)


def root_normalize(P: np.ndarray, joints: List[str]) -> np.ndarray:
    """Hip-root 중심화 + 토르소 길이 정규화"""
    h = joints.index("Hip")
    P = P - P[:, [h], :]
    if "Neck" in joints:
        n = joints.index("Neck")
        scale = np.linalg.norm(P[:, n] - P[:, h], axis=1).mean()
        P = P / max(scale, 1e-8)
    return P


def pose_verdict_for_frame(angles: Dict[str, float], target_angles: Dict[str, float],
                           tolerance: float = 15.0) -> bool:
    """한 프레임이 정답 자세의 허용범위 안에 들어가는지 판정 (O/X)"""
    for k, target in target_angles.items():
        if abs(angles[k] - target) > tolerance:
            return False
    return True


# Pose target — angle_scorer의 핵심 각도 정답값과 일치
POSE_TARGETS = {
    "the_seal": {"hip": 80.73, "knee": 35.64},
    "bridging": {"hip": 170.0, "knee": 90.0},
    "spine_stretch": {"hip": 80.0, "knee": 175.0},
}


def smooth_dataframe(df: pd.DataFrame, joints: List[str], window: int = 11, poly: int = 3) -> pd.DataFrame:
    """3D 좌표 컬럼에 Savitzky-Golay 스무딩"""
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return df
    df = df.copy()
    w = window if window % 2 == 1 else window + 1
    if len(df) < w + 1:
        return df
    for j in joints:
        for a in ("x", "y", "z"):
            col = f"{j}_{a}"
            if col in df.columns:
                df[col] = savgol_filter(df[col].to_numpy(dtype=float), w, poly, mode="interp")
    return df


def evaluate(pred_csv: Path, gt_csv: Path, angle_thresholds: List[float] = (10, 15, 20),
             joint_thresholds: List[float] = (0.10, 0.15, 0.20),
             onpose_strict_tol_deg: float = 15.0,
             pose_key: str = "the_seal",
             smooth_gt: bool = False,
             smooth_pred: bool = False,
             smooth_window: int = 11) -> Dict:
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    n = min(len(pred), len(gt))
    pred = pred.iloc[:n].reset_index(drop=True)
    gt = gt.iloc[:n].reset_index(drop=True)
    if smooth_gt:
        gt = smooth_dataframe(gt, JOINTS, window=smooth_window)
    if smooth_pred:
        pred = smooth_dataframe(pred, JOINTS, window=smooth_window)

    # -------- 1) Angle PCK --------
    pa = compute_angles(pred)
    ga = compute_angles(gt)
    angle_errors = {k: np.abs(pa[k] - ga[k]) for k in pa.keys()}
    angle_errors["hip"] = (angle_errors["hip_left"] + angle_errors["hip_right"]) / 2.0
    angle_errors["knee"] = (angle_errors["knee_left"] + angle_errors["knee_right"]) / 2.0

    angle_pck = {}
    angle_mae = {k: float(np.mean(v)) for k, v in angle_errors.items()}
    for tau in angle_thresholds:
        angle_pck[f"@{int(tau)}deg"] = {
            "hip": float(np.mean(angle_errors["hip"] <= tau) * 100),
            "knee": float(np.mean(angle_errors["knee"] <= tau) * 100),
            "trunk": float(np.mean(angle_errors["trunk"] <= tau) * 100),
            "avg": float(np.mean([np.mean(angle_errors[k] <= tau) for k in ("hip", "knee", "trunk")]) * 100),
        }

    # -------- 2) Joint PCK (root-normalized) --------
    joints_present = [j for j in JOINTS if all(f"{j}_{a}" in pred.columns and f"{j}_{a}" in gt.columns
                                                for a in ("x", "y", "z"))]
    P = root_normalize(stack_xyz(pred, joints_present), joints_present)
    G = root_normalize(stack_xyz(gt, joints_present), joints_present)
    err = np.linalg.norm(P - G, axis=-1)        # (n_frames, n_joints)
    mpjpe = float(err.mean())
    joint_pck = {}
    for tau in joint_thresholds:
        joint_pck[f"@{tau:.2f}"] = float(np.mean(err <= tau) * 100)

    # -------- 3) OnPose-strict per-frame accuracy --------
    # 한 프레임이 "정답"으로 인정되려면: hip, knee, trunk 셋 다 ±15° 이내
    per_frame_ok = (
        (angle_errors["hip"] <= onpose_strict_tol_deg) &
        (angle_errors["knee"] <= onpose_strict_tol_deg) &
        (angle_errors["trunk"] <= onpose_strict_tol_deg)
    )
    onpose_accuracy = float(per_frame_ok.mean() * 100)

    # 핵심 각도만 (hip + knee, trunk 제외 — 멘토링 가중치 반영)
    per_frame_core_ok = (
        (angle_errors["hip"] <= onpose_strict_tol_deg) &
        (angle_errors["knee"] <= onpose_strict_tol_deg)
    )
    onpose_core_accuracy = float(per_frame_core_ok.mean() * 100)

    # -------- 4) Pose Verdict Agreement on END-phase (자세 유지 구간만) --------
    # GT 시퀀스에는 동작 전후가 모두 포함되어 있어 정답 각도와 거리가 먼 프레임이 많음.
    # 자세 유지(END) 구간만 추출해서 GT/Pred 판정 일치율을 계산하는 게 의미 있음.
    target = POSE_TARGETS.get(pose_key, POSE_TARGETS["the_seal"])
    pred_hip = (pa["hip_left"] + pa["hip_right"]) / 2.0
    pred_knee = (pa["knee_left"] + pa["knee_right"]) / 2.0
    gt_hip = (ga["hip_left"] + ga["hip_right"]) / 2.0
    gt_knee = (ga["knee_left"] + ga["knee_right"]) / 2.0

    # END-phase 마스크 (pred에 phase_idx 컬럼이 있으면 사용, 없으면 전체)
    if "phase_idx" in pred.columns:
        end_mask = pred["phase_idx"].to_numpy() == 2   # progress3: 2 = END
    else:
        end_mask = np.ones(n, dtype=bool)
    n_end = int(end_mask.sum())

    verdict_tols = [15.0, 20.0]
    verdict_agreement = {}
    for tol in verdict_tols:
        # 1) Angle Agreement (GT와 Pred 각도 자체의 차이)
        agree_hip = (np.abs(pred_hip - gt_hip)[end_mask] <= tol).mean() * 100 if n_end else 0.0
        agree_knee = (np.abs(pred_knee - gt_knee)[end_mask] <= tol).mean() * 100 if n_end else 0.0
        # 2) Verdict Agreement (정답 자세 ±tol 판정 일치)
        pred_ok = (np.abs(pred_hip - target["hip"]) <= tol) & (np.abs(pred_knee - target["knee"]) <= tol)
        gt_ok = (np.abs(gt_hip - target["hip"]) <= tol) & (np.abs(gt_knee - target["knee"]) <= tol)
        if n_end:
            agreement = float((pred_ok[end_mask] == gt_ok[end_mask]).mean() * 100)
        else:
            agreement = 0.0
        verdict_agreement[f"tol_{int(tol)}deg"] = {
            "angle_agreement_hip_pct":  round(float(agree_hip), 2),
            "angle_agreement_knee_pct": round(float(agree_knee), 2),
            "verdict_agreement_pct":    round(agreement, 2),
            "n_end_frames": n_end,
        }

    summary = {
        "pred_csv": str(pred_csv),
        "gt_csv": str(gt_csv),
        "pose_key": pose_key,
        "frames": n,
        "joints_used": joints_present,
        "angle_mae_deg": {k: round(v, 3) for k, v in angle_mae.items()},
        "angle_pck": angle_pck,
        "joint_pck_root_norm": joint_pck,
        "mpjpe_root_norm": round(mpjpe, 4),
        "onpose_accuracy_pct_strict_15deg_all3": round(onpose_accuracy, 2),
        "onpose_accuracy_pct_core_hip_knee": round(onpose_core_accuracy, 2),
        "pose_verdict_agreement": verdict_agreement,
    }
    return summary


def pretty_print(summary: Dict) -> None:
    deg = "deg"
    line = "=" * 76
    print(line)
    print(f"  2D->3D Lifting Accuracy Report  (pose={summary['pose_key']})")
    print(line)
    print(f"  Frames compared : {summary['frames']}")
    print(f"  MPJPE (root-norm): {summary['mpjpe_root_norm']:.4f}")
    print()
    print(f"  Angle MAE ({deg}):")
    for k, v in summary["angle_mae_deg"].items():
        print(f"    - {k:<14} {v:>7.2f}{deg}")
    print()
    print(f"  Angle PCK (오차 임계값 이내 프레임 비율):")
    for thr_key, vals in summary["angle_pck"].items():
        marker = "  [80%+]" if vals["hip"] >= 80.0 else ""
        print(f"    err {thr_key:<8}   hip {vals['hip']:>6.1f}%  knee {vals['knee']:>6.1f}%  "
              f"trunk {vals['trunk']:>6.1f}%   avg {vals['avg']:>6.1f}%{marker}")
    print()
    print(f"  Pose Verdict Agreement on END-phase  (자세 유지 구간만):")
    for tol_key, vals in summary["pose_verdict_agreement"].items():
        m = "  [PASS]" if vals["verdict_agreement_pct"] >= 80.0 else ""
        print(f"    {tol_key:<10} n_end={vals['n_end_frames']:>4}  "
              f"hip-agree {vals['angle_agreement_hip_pct']:>6.2f}%  "
              f"knee-agree {vals['angle_agreement_knee_pct']:>6.2f}%  "
              f"verdict {vals['verdict_agreement_pct']:>6.2f}%{m}")
    print()
    print(f"  OnPose Accuracy (per-frame strict):")
    print(f"    All-3 ({summary.get('onpose_accuracy_pct_strict_15deg_all3', 0):>6.2f}%, hip+knee+trunk all <= 15deg)")
    print(f"    Core  ({summary['onpose_accuracy_pct_core_hip_knee']:>6.2f}%, hip+knee only)")
    print(line)
    # 합격 판정 (가장 직접적인 지표)
    headline_hip15 = summary["angle_pck"]["@15deg"]["hip"]
    headline_knee20 = summary["angle_pck"]["@20deg"]["knee"]
    headline_hip20 = summary["angle_pck"]["@20deg"]["hip"]
    v15 = summary["pose_verdict_agreement"]["tol_15deg"]["verdict_agreement_pct"]
    v20 = summary["pose_verdict_agreement"]["tol_20deg"]["verdict_agreement_pct"]
    print(f"  HEADLINE METRICS (멘토링 목표: 80%+ / 90%+):")
    print(f"    [Hip  PCK @15deg]                 {headline_hip15:>6.2f}%   {'PASS80' if headline_hip15 >= 80 else 'FAIL'} {'PASS90' if headline_hip15 >= 90 else ''}")
    print(f"    [Hip  PCK @20deg]                 {headline_hip20:>6.2f}%   {'PASS80' if headline_hip20 >= 80 else 'FAIL'} {'PASS90' if headline_hip20 >= 90 else ''}")
    print(f"    [Knee PCK @20deg]                 {headline_knee20:>6.2f}%   {'PASS80' if headline_knee20 >= 80 else 'FAIL'} {'PASS90' if headline_knee20 >= 90 else ''}")
    print(f"    [End-phase Verdict Agreement @15] {v15:>6.2f}%   {'PASS80' if v15 >= 80 else 'FAIL'} {'PASS90' if v15 >= 90 else ''}")
    print(f"    [End-phase Verdict Agreement @20] {v20:>6.2f}%   {'PASS80' if v20 >= 80 else 'FAIL'} {'PASS90' if v20 >= 90 else ''}")
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="2D→3D Lifting Accuracy (OnPose v6)")
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tol-deg", type=float, default=15.0,
                    help="OnPose-strict 정확도용 ±tolerance (default 15deg)")
    ap.add_argument("--pose", type=str, default="the_seal",
                    choices=list(POSE_TARGETS.keys()),
                    help="자세 종류 (verdict agreement용 정답 각도 선택)")
    ap.add_argument("--smooth-gt", action="store_true",
                    help="GT 키포인트 스무딩 (Savitzky-Golay, GT 자체 jitter가 클 때)")
    ap.add_argument("--smooth-pred", action="store_true",
                    help="Pred 키포인트 스무딩 (공정 비교)")
    ap.add_argument("--smooth-window", type=int, default=11)
    args = ap.parse_args()

    summary = evaluate(args.pred_csv, args.gt_csv,
                       onpose_strict_tol_deg=args.tol_deg, pose_key=args.pose,
                       smooth_gt=args.smooth_gt, smooth_pred=args.smooth_pred,
                       smooth_window=args.smooth_window)
    pretty_print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
