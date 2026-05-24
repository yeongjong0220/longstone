"""
AI Hub 필라테스 데이터셋(3D keypoint CSV + JSON)에서
자세별 정답 각도 분포를 추출 → rubric 갱신 근거 자료.

데이터 구조 예:
  Mat_train_1_40/Spine Stretch/중급/actorP045/20220913_11.33.56_가산A/
    ├── keypoints_..._actorP045_..._.csv          ← 3D ground truth
    └── normal_AI_..._.json                        ← start_frame / end_frame

JSON에서 동작 구간(START→MIDDLE→END)을 가져와 END phase 프레임만 사용.

사용:
  python eval/extract_pose_stats.py \
    --root "D:/dataset/216.필라테스 동작 데이터/01-1.정식개방데이터/Training_1/Mat" \
    --out reports/pose_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

POSE_ALIASES = {
    "Spine Stretch": "Spine_Stretch",
    "The Seal": "The_Seal",
    "Bridging": "Bridging",
}


def vec(df, name):
    cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)


def angle(a, b, c):
    u = a - b; v = c - b
    u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-8, None)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1), -1, 1)))


def compute_angles(df) -> Dict[str, np.ndarray]:
    lhip = vec(df, "LHip"); rhip = vec(df, "RHip")
    lknee = vec(df, "LKnee"); rknee = vec(df, "Rknee")
    lankle = vec(df, "LAnkle"); rankle = vec(df, "RAnkle")
    neck = vec(df, "Neck"); hip = vec(df, "Hip"); head = vec(df, "Head")
    if any(x is None for x in [lhip, rhip, lknee, rknee, lankle, rankle, neck, hip, head]):
        return {}
    return {
        "hip_left": angle(neck, lhip, lknee),
        "hip_right": angle(neck, rhip, rknee),
        "knee_left": angle(lhip, lknee, lankle),
        "knee_right": angle(rhip, rknee, rankle),
        "trunk": angle(hip, neck, head),
    }


def trim_with_json(df: pd.DataFrame, jpath: Path) -> pd.DataFrame:
    try:
        j = json.loads(jpath.read_text(encoding="utf-8"))
        ann = j.get("annotations", j)
        s = int(ann.get("start_frame", 0))
        e = int(ann.get("end_frame", len(df)))
        return df.iloc[s:e + 1].reset_index(drop=True)
    except Exception:
        return df


def find_3d_csvs(root: Path) -> List[Dict]:
    """루트 하위 모든 actor 디렉터리에서 (pose_kr, csv_path, json_path) 추출"""
    out = []
    for pose_dir_kr in POSE_ALIASES.keys():
        # 두 가지 위치를 모두 탐색
        candidates = list(root.rglob(pose_dir_kr))
        for cand in candidates:
            if not cand.is_dir():
                continue
            # actorPxxx 디렉터리들 순회
            for actor in cand.rglob("actorP*"):
                if not actor.is_dir():
                    continue
                # actor 안에 timestamp_가산A 폴더
                for ts_dir in actor.iterdir():
                    if not ts_dir.is_dir():
                        continue
                    # 그 안에 actor 이름이 들어간 csv 파일 (3D GT)
                    for csv in ts_dir.glob("*.csv"):
                        # camera 디렉터리가 아닌, 최상위 csv만 (3D GT)
                        if "camera" in str(csv.parent.name).lower():
                            continue
                        # 매칭되는 json 찾기
                        jsons = list(ts_dir.glob("normal_AI_*.json")) + list(ts_dir.glob("*.json"))
                        out.append({
                            "pose": POSE_ALIASES[pose_dir_kr],
                            "pose_kr": pose_dir_kr,
                            "csv": csv,
                            "json": jsons[0] if jsons else None,
                            "actor": actor.name,
                        })
                        break   # 한 timestamp당 하나
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] /
                                                "reports" / "pose_stats.json")
    ap.add_argument("--limit-per-pose", type=int, default=20,
                    help="자세당 최대 actor 수 (속도용)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[error] root not found: {args.root}")
        return 1

    print(f"[scan] {args.root}")
    csvs = find_3d_csvs(args.root)
    print(f"[scan] found {len(csvs)} actor sequences")
    if not csvs:
        return 1

    by_pose = defaultdict(list)
    for c in csvs:
        by_pose[c["pose"]].append(c)

    results: Dict[str, Dict] = {}
    for pose_key, items in by_pose.items():
        print(f"\n[{pose_key}] {len(items)} actor sequences")
        items = items[:args.limit_per_pose]
        all_angles: Dict[str, List[float]] = defaultdict(list)
        n_used = 0
        for item in items:
            try:
                df = pd.read_csv(item["csv"])
                if item["json"]:
                    df = trim_with_json(df, item["json"])
                angles = compute_angles(df)
                if not angles:
                    continue
                # END phase (자세 유지) 만 — 동작 중간 한가운데 30~70% 구간 사용
                n = len(df)
                if n < 30:
                    continue
                lo, hi = int(n * 0.3), int(n * 0.7)
                for k, arr in angles.items():
                    all_angles[k].extend(arr[lo:hi].tolist())
                # 좌우 평균
                hip_avg = (angles["hip_left"][lo:hi] + angles["hip_right"][lo:hi]) / 2.0
                knee_avg = (angles["knee_left"][lo:hi] + angles["knee_right"][lo:hi]) / 2.0
                all_angles["hip"].extend(hip_avg.tolist())
                all_angles["knee"].extend(knee_avg.tolist())
                n_used += 1
            except Exception as exc:
                print(f"  skip {item['csv'].name}: {exc}")
                continue

        results[pose_key] = {"n_actors": n_used, "stats": {}, "velocity_stats": {}}
        print(f"  used {n_used} sequences")
        for k in ["hip", "knee", "trunk"]:
            if k in all_angles and all_angles[k]:
                arr = np.asarray(all_angles[k])
                stats = {
                    "n": int(len(arr)),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                }
                results[pose_key]["stats"][k] = stats
                # 각속도 = frame-to-frame 변화량의 절댓값 분포
                vel = np.abs(np.diff(arr))
                results[pose_key]["velocity_stats"][k] = {
                    "n": int(len(vel)),
                    "mean_dps": float(vel.mean()),       # degrees per frame
                    "std_dps": float(vel.std()),
                    "median_dps": float(np.median(vel)),
                    "p75_dps": float(np.percentile(vel, 75)),
                    "p95_dps": float(np.percentile(vel, 95)),
                }
                v = results[pose_key]["velocity_stats"][k]
                print(f"  {k:<6}  mean={stats['mean']:.2f}°  std={stats['std']:.2f}  "
                      f"|  velocity mean={v['mean_dps']:.2f}°/f  p95={v['p95_dps']:.2f}°/f")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] saved -> {args.out}")

    # 현재 rubric과 비교
    print("\n[rubric comparison]")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.angle_scorer import RUBRICS
    for pose_key, data in results.items():
        if pose_key not in RUBRICS:
            continue
        rub = RUBRICS[pose_key]
        print(f"\n  {pose_key}:")
        for k, spec in rub.angles.items():
            if k in data["stats"]:
                s = data["stats"][k]
                delta = s["mean"] - spec.target_deg
                tol_iqr = (s["p75"] - s["p25"]) / 2.0
                print(f"    {k:<6}  rubric_target={spec.target_deg:.2f}°  "
                      f"data_mean={s['mean']:.2f}°  (Δ{delta:+.2f}°)  "
                      f"recommended_tolerance={max(tol_iqr, 8.0):.1f}° (current {spec.tolerance_deg:.1f}°)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
