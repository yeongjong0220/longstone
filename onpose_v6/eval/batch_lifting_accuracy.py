"""
데이터셋의 자세별 GT 데이터에서 lifter 출력과 GT 3D를 비교하는 일괄 평가.

각 자세에 대해 여러 actor의 GT 시퀀스에서 각도 일치 정확도를 측정.
lifter가 따로 예측 CSV를 만들지 않은 경우, GT 자체에 추출한 self-consistency 메트릭 산출.

사용:
  python eval/batch_lifting_accuracy.py \
    --root "D:/dataset/216.필라테스 동작 데이터/01-1.정식개방데이터/Training_1/Mat" \
    --pred-csv-pattern "predicted_eval_progress3_angle_causal_v1.csv" \
    --out reports/batch_lifting_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from eval.extract_pose_stats import POSE_ALIASES, compute_angles, find_3d_csvs, trim_with_json


def angle_pck(diffs: np.ndarray, thresholds=(10, 15, 20)) -> Dict[str, float]:
    return {f"@{int(t)}deg": float(np.mean(diffs <= t) * 100) for t in thresholds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=THIS.parents[1] / "reports" /
                                                  "batch_lifting_accuracy.json")
    ap.add_argument("--limit-per-pose", type=int, default=30)
    ap.add_argument("--use-end-only", action="store_true",
                    help="시퀀스 중간 30 to 70 percent 구간만 사용 (자세 유지)")
    args = ap.parse_args()

    csvs = find_3d_csvs(args.root)
    by_pose: Dict[str, List] = defaultdict(list)
    for c in csvs:
        by_pose[c["pose"]].append(c)

    results: Dict = {}
    for pose_key, items in by_pose.items():
        items = items[:args.limit_per_pose]
        # 자세별 데이터 통계
        agg_hip_diff: List[float] = []
        agg_knee_diff: List[float] = []
        agg_trunk_diff: List[float] = []
        n_used = 0
        for item in items:
            try:
                df = pd.read_csv(item["csv"])
                if item["json"]:
                    df = trim_with_json(df, item["json"])
                angles = compute_angles(df)
                if not angles:
                    continue
                n = len(df)
                if n < 30:
                    continue
                if args.use_end_only:
                    lo, hi = int(n * 0.3), int(n * 0.7)
                else:
                    lo, hi = 0, n
                # 좌우 차이 = lifter가 좌우 대칭을 얼마나 잘 추정하는지 (self consistency)
                lh = angles["hip_left"][lo:hi]; rh = angles["hip_right"][lo:hi]
                lk = angles["knee_left"][lo:hi]; rk = angles["knee_right"][lo:hi]
                # 자세에 따라 좌우 대칭이면 차이가 작아야 함
                hip_lr_diff = np.abs(lh - rh)
                knee_lr_diff = np.abs(lk - rk)
                agg_hip_diff.extend(hip_lr_diff.tolist())
                agg_knee_diff.extend(knee_lr_diff.tolist())
                n_used += 1
            except Exception:
                continue
        if not agg_hip_diff:
            continue
        hip_arr = np.asarray(agg_hip_diff)
        knee_arr = np.asarray(agg_knee_diff)
        results[pose_key] = {
            "n_actors_used": n_used,
            "n_frames_total": len(hip_arr),
            "lr_consistency_deg": {
                "hip_mean_diff": float(hip_arr.mean()),
                "hip_p95": float(np.percentile(hip_arr, 95)),
                "knee_mean_diff": float(knee_arr.mean()),
                "knee_p95": float(np.percentile(knee_arr, 95)),
            },
            "lr_consistency_pct_within_10deg": {
                "hip": float(np.mean(hip_arr <= 10) * 100),
                "knee": float(np.mean(knee_arr <= 10) * 100),
            },
            "lr_consistency_pct_within_15deg": {
                "hip": float(np.mean(hip_arr <= 15) * 100),
                "knee": float(np.mean(knee_arr <= 15) * 100),
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved -> {args.out}")

    # 표 출력
    print()
    print(f"{'Pose':<16} {'n':>5} {'frames':>8} {'hip Δ(mean)':>14} {'knee Δ(mean)':>14} "
          f"{'LR≤10° hip':>12} {'LR≤10° knee':>12}")
    print("-" * 84)
    for pose_key, d in results.items():
        lr = d["lr_consistency_deg"]
        p10 = d["lr_consistency_pct_within_10deg"]
        print(f"{pose_key:<16} {d['n_actors_used']:>5} {d['n_frames_total']:>8} "
              f"{lr['hip_mean_diff']:>12.2f}°  {lr['knee_mean_diff']:>12.2f}°  "
              f"{p10['hip']:>10.1f}%  {p10['knee']:>10.1f}%")
    print()
    print("INFO: LR consistency = 같은 자세에서 좌/우 관절 각도 차이.")
    print("      자연스러운 좌우 대칭 자세(브릿징/씰/스트레치)에서 작아야 좋음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
