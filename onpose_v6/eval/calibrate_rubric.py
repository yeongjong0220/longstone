"""
extract_pose_stats.py 결과 (reports/pose_stats.json)를 기반으로
데이터 기반 rubric을 자동 생성해 reports/rubric_calibrated.json 으로 저장.

생성 규칙:
- target_deg = data_mean (또는 median)
- tolerance_deg = max(8.0, std * 1.5)  — 자세별 자연 분산 반영
- weight, importance, name_kr은 기존 rubric 유지

사용:
  python eval/calibrate_rubric.py
  python eval/calibrate_rubric.py --use median
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from core.angle_scorer import RUBRICS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path,
                    default=THIS.parents[1] / "reports" / "pose_stats.json")
    ap.add_argument("--out", type=Path,
                    default=THIS.parents[1] / "reports" / "rubric_calibrated.json")
    ap.add_argument("--use", choices=["mean", "median"], default="median",
                    help="target_deg 산출 방법")
    ap.add_argument("--min-tol", type=float, default=8.0)
    ap.add_argument("--std-mult", type=float, default=1.5)
    ap.add_argument("--min-actors", type=int, default=3,
                    help="이 수 미만이면 기존 rubric 유지 (통계 부족)")
    args = ap.parse_args()

    if not args.stats.exists():
        print(f"[error] stats not found: {args.stats}")
        print("  먼저: python eval/extract_pose_stats.py --root <dataset_root>")
        return 1
    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    new_rubric = {}
    for pose_key, rub in RUBRICS.items():
        new_rubric[pose_key] = {
            "name_kr": rub.pose_name_kr,
            "pass_threshold": rub.pass_threshold,
            "angles": {},
        }
        data = stats.get(pose_key, {})
        n_actors = data.get("n_actors", 0)
        s_stats = data.get("stats", {})
        if n_actors < args.min_actors:
            # 통계 부족 → 기존 rubric 그대로
            for k, spec in rub.angles.items():
                new_rubric[pose_key]["angles"][k] = {
                    "name_kr": spec.name,
                    "target_deg": spec.target_deg,
                    "tolerance_deg": spec.tolerance_deg,
                    "weight": spec.weight,
                    "importance": spec.importance,
                    "source": f"unchanged (n_actors={n_actors} < {args.min_actors})",
                }
            new_rubric[pose_key]["n_actors_used"] = n_actors
            continue

        for k, spec in rub.angles.items():
            s = s_stats.get(k, {})
            if not s:
                new_rubric[pose_key]["angles"][k] = {
                    "name_kr": spec.name,
                    "target_deg": spec.target_deg,
                    "tolerance_deg": spec.tolerance_deg,
                    "weight": spec.weight,
                    "importance": spec.importance,
                    "source": "unchanged (no data for this angle)",
                }
                continue
            tgt = s.get(args.use, s.get("mean"))
            tol = max(args.min_tol, s.get("std", 8.0) * args.std_mult)
            new_rubric[pose_key]["angles"][k] = {
                "name_kr": spec.name,
                "target_deg": round(float(tgt), 2),
                "tolerance_deg": round(float(tol), 2),
                "weight": spec.weight,
                "importance": spec.importance,
                "source": f"calibrated (n={s.get('n', 0)}, mean={s['mean']:.2f}, std={s['std']:.2f})",
                "old_target_deg": spec.target_deg,
                "old_tolerance_deg": spec.tolerance_deg,
            }
        new_rubric[pose_key]["n_actors_used"] = n_actors

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(new_rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved -> {args.out}")

    # 비교 출력
    print()
    print(f"{'Pose':<16} {'Angle':<6} {'Old target':>12} {'New target':>12} {'Δ':>8} {'Old tol':>10} {'New tol':>10}")
    print("-" * 80)
    for pkey, val in new_rubric.items():
        for ak, ang in val["angles"].items():
            if "old_target_deg" not in ang:
                continue
            old = ang["old_target_deg"]
            new = ang["target_deg"]
            print(f"{pkey:<16} {ak:<6} {old:>11.2f}° {new:>11.2f}° {new-old:>+7.1f}° "
                  f"{ang['old_tolerance_deg']:>9.1f}° {ang['tolerance_deg']:>9.1f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
