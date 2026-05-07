"""input 그리드 합성기.

자세 × 프레임수 × warn_or_fail_ratio × 우세 phase × 우세 feature × max_d2 severity 조합으로
약 500개의 v4 분석 요약 dict 를 결정적으로 생성하고, 그것을 v4 코드의
`format_summary_for_prompt` 로 직렬화해 inputs.jsonl 로 떨군다.

런타임(`live_ai_coach_v4_quality.py`)이 `summarize_score_rows` 또는 `fallback_summary` 가 만든
딕셔너리를 그대로 `format_summary_for_prompt` 에 넘기는 것과 동일한 경로를 따라가도록 했다.
직렬화 함수는 v4 번들에서 직접 import 한다 (numpy/pandas 만 의존, cv2 없음).

Usage:
    python build_inputs.py --out data/synth/inputs.jsonl --seed-out data/synth/synth_seed.json
    python build_inputs.py --limit 10 --out data/synth/inputs_dryrun.jsonl ...   # dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- v4 직렬화 함수 import ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]  # min/llm_finetune/data/synth → repo root
LIFTER_DIR = PROJECT_ROOT / "live_ai_coach_v4_bundle" / "pilates_temporal_lifter"
if not LIFTER_DIR.exists():
    raise RuntimeError(f"Cannot find lifter dir: {LIFTER_DIR}")
sys.path.insert(0, str(LIFTER_DIR))
from runtime_quality import format_summary_for_prompt  # noqa: E402

# --- pose 메타 ---
sys.path.insert(0, str(SCRIPT_DIR))
from pose_meta import (  # noqa: E402
    ALL_POSES,
    BRIDGING,
    SHARED_INSTRUCTION,
    SPINE_STRETCH,
    THE_SEAL,
    PoseMeta,
)

# --- 그리드 축 ---
FRAMES_AXIS = [30, 45, 60, 90, 120]
RATIO_AXIS = [0.0, 0.05, 0.15, 0.30, 0.50]
SEVERITY_AXIS = [0.5, 1.5, 3.0, 5.0]  # max_mahalanobis_d2 대표값

# warn vs fail 분배 비율 (severity 가 클수록 fail 우세)
WARN_FAIL_SPLIT = {
    0.5: (1.0, 0.0),
    1.5: (0.85, 0.15),
    3.0: (0.55, 0.45),
    5.0: (0.30, 0.70),
}


def _pick_dom_phase(pose: PoseMeta, idx: int) -> str | None:
    """phase_names 가 있으면 결정적으로 하나 고름. 없으면 None (Spine Stretch)."""
    if not pose.phase_names:
        return None
    return pose.phase_names[idx % len(pose.phase_names)]


def _pick_top_feature(pose: PoseMeta, idx: int) -> str | None:
    if pose.has_scorer:
        return pose.feature_names[idx % len(pose.feature_names)]
    if pose.fallback_features:
        return pose.fallback_features[idx % len(pose.fallback_features)]
    return None


def _split_status_counts(frames: int, ratio: float, severity: float) -> Dict[str, int]:
    """frames 와 ratio 를 PASS/WARN/FAIL 카운트로 분배."""
    n_issue = int(round(frames * ratio))
    if n_issue == 0:
        return {"PASS": frames}
    warn_share, fail_share = WARN_FAIL_SPLIT[severity]
    n_fail = int(round(n_issue * fail_share))
    n_warn = n_issue - n_fail
    out = {"PASS": frames - n_issue}
    if n_warn > 0:
        out["WARN"] = n_warn
    if n_fail > 0:
        out["FAIL"] = n_fail
    return out


def _phase_counts(pose: PoseMeta, frames: int, dom_phase: str | None) -> Dict[str, int]:
    if not pose.phase_names:
        return {}
    base = frames // len(pose.phase_names)
    counts = {p: base for p in pose.phase_names}
    leftover = frames - base * len(pose.phase_names)
    if dom_phase is not None and dom_phase in counts:
        counts[dom_phase] += leftover + base // 2
        # 다른 phase 에서 base//2 만큼 빌려옴
        debt = base // 2
        for p in pose.phase_names:
            if p == dom_phase or debt <= 0:
                continue
            give = min(debt, counts[p] - 1)
            counts[p] -= give
            debt -= give
    else:
        # 잔여를 첫 phase 에 몰아줌
        counts[pose.phase_names[0]] += leftover
    # 0 이하 방지
    return {k: max(int(v), 1 if k == dom_phase else 0) for k, v in counts.items() if v > 0}


def _top_issue_counts(pose: PoseMeta, n_issue: int, top_feature: str | None) -> Dict[str, int]:
    if n_issue == 0 or top_feature is None:
        return {}
    pool = pose.feature_names if pose.has_scorer else pose.fallback_features
    if top_feature not in pool:
        return {}
    counts = {top_feature: int(round(n_issue * 0.7))}
    others = [f for f in pool if f != top_feature]
    remaining = n_issue - counts[top_feature]
    for i, f in enumerate(others):
        if remaining <= 0:
            break
        share = max(1, remaining // (len(others) - i))
        counts[f] = share
        remaining -= share
    return counts


def _worst_frames_scorer(
    pose: PoseMeta,
    frames: int,
    dom_phase: str | None,
    top_feature: str | None,
    severity: float,
    rng: random.Random,
) -> List[Dict]:
    """scorer 케이스 worst_frames: phase_name + status + d2 + top_feature + z."""
    if not pose.has_scorer or top_feature is None:
        return []
    out = []
    n = min(5, max(2, frames // 12))
    for i in range(n):
        d2 = severity * (1.0 - i * 0.12)
        status = "FAIL" if d2 > severity * 0.7 else "WARN"
        z = round((d2 / max(severity, 0.1)) * (rng.uniform(1.5, 3.0)), 2)
        out.append(
            {
                "frame": rng.randint(1, max(2, frames - 1)),
                "phase_name": dom_phase if dom_phase and rng.random() < 0.7 else rng.choice(pose.phase_names),
                "status": status,
                "mahalanobis_d2": round(d2, 3),
                "top_feature": top_feature if rng.random() < 0.7 else rng.choice(pose.feature_names),
                "top_feature_z": z if rng.random() < 0.5 else -z,
            }
        )
    return sorted(out, key=lambda r: -r["mahalanobis_d2"])


def _worst_frames_fallback(
    pose: PoseMeta,
    frames: int,
    n_issue: int,
    top_feature: str | None,
    rng: random.Random,
) -> List[Dict]:
    """fallback 케이스 worst_frames: live_ai_coach_v4_quality.py 의 fallback_summary 와 동일 스키마."""
    if pose.has_scorer or n_issue == 0 or top_feature is None:
        return []
    refs = {"hip": 80.0, "knee": 175.0}  # Spine Stretch 기준
    out = []
    for i in range(min(n_issue, 8)):
        ref = refs.get(top_feature, 90.0)
        deviation = rng.uniform(8.0, 25.0) * (1 if rng.random() < 0.5 else -1)
        out.append(
            {
                "frame": rng.randint(1, max(2, frames - 1)),
                "feature": top_feature,
                "value": round(ref + deviation, 2),
                "ref": ref,
            }
        )
    return out


def synth_summary(
    pose: PoseMeta,
    frames: int,
    ratio: float,
    dom_phase: str | None,
    top_feature: str | None,
    severity: float,
    rng: random.Random,
) -> Dict:
    status_counts = _split_status_counts(frames, ratio, severity)
    n_issue = sum(v for k, v in status_counts.items() if k != "PASS")
    phase_counts = _phase_counts(pose, frames, dom_phase) if pose.has_scorer else {}
    top_issue = _top_issue_counts(pose, n_issue, top_feature)

    if pose.has_scorer:
        max_d2 = severity if n_issue > 0 else round(severity * 0.4, 3)
        mean_d2 = round(max_d2 * (0.30 + 0.20 * (n_issue / max(frames, 1))), 3)
        worst = _worst_frames_scorer(pose, frames, dom_phase, top_feature, max_d2, rng)
    else:
        max_d2 = 0.0
        mean_d2 = 0.0
        worst = _worst_frames_fallback(pose, frames, n_issue, top_feature, rng)

    return {
        "frames": int(frames),
        "status_counts": status_counts,
        "phase_counts": phase_counts,
        "top_issue_counts": top_issue,
        "mean_mahalanobis_d2": float(mean_d2),
        "max_mahalanobis_d2": float(max_d2),
        "warn_or_fail_ratio": round(n_issue / max(frames, 1), 3),
        "worst_frames": worst,
    }


def _enumerate_combinations(pose: PoseMeta) -> List[Tuple]:
    """deterministic 조합 enumeration."""
    # PASS-only 케이스용으로 ratio=0.0 따로 처리
    combos: List[Tuple] = []
    phases = pose.phase_names if pose.phase_names else [None]
    features_pool = pose.feature_names if pose.has_scorer else pose.fallback_features
    if not features_pool:
        features_pool = [None]

    for frames in FRAMES_AXIS:
        for ratio in RATIO_AXIS:
            for severity in SEVERITY_AXIS:
                if ratio == 0.0:
                    # PASS-only — phase·feature 무의미
                    combos.append((frames, ratio, None, None, SEVERITY_AXIS[0]))
                    continue
                for dom_phase in phases:
                    for top_feature in features_pool:
                        combos.append((frames, ratio, dom_phase, top_feature, severity))
    # PASS-only 중복 제거
    seen = set()
    deduped = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _stride_subsample(combos: List[Tuple], target: int, rng: random.Random) -> List[Tuple]:
    if len(combos) <= target:
        return combos
    # 균등 stride 샘플링
    stride = len(combos) / target
    picked = [combos[int(i * stride)] for i in range(target)]
    # 중복 제거 후 부족하면 random fill
    seen = set(picked)
    dedup = list(seen)
    if len(dedup) < target:
        remaining = [c for c in combos if c not in seen]
        rng.shuffle(remaining)
        dedup.extend(remaining[: target - len(dedup)])
    return dedup[:target]


def build_for_pose(pose: PoseMeta, rng: random.Random) -> List[Dict]:
    combos = _enumerate_combinations(pose)
    rng.shuffle(combos)  # deterministic given seeded rng
    combos = _stride_subsample(combos, pose.target_count, rng)
    out = []
    for idx, (frames, ratio, dom_phase, top_feature, severity) in enumerate(combos):
        sub_rng = random.Random(rng.randint(0, 1 << 30) + idx)
        summary = synth_summary(pose, frames, ratio, dom_phase, top_feature, severity, sub_rng)
        input_text = format_summary_for_prompt(pose.name_en, summary)
        out.append(
            {
                "id": f"{pose.key}__{idx:04d}",
                "pose_key": pose.key,
                "pose_name_en": pose.name_en,
                "pose_name_kr": pose.name_kr,
                "scorer_available": pose.has_scorer,
                "grid": {
                    "frames": frames,
                    "ratio": ratio,
                    "dom_phase": dom_phase,
                    "top_feature": top_feature,
                    "severity": severity,
                },
                "summary": summary,
                "instruction": SHARED_INSTRUCTION,
                "input": input_text,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR.parent / "synth" / "inputs.jsonl")
    ap.add_argument("--seed-out", type=Path, default=SCRIPT_DIR.parent / "synth" / "synth_seed.json")
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--limit", type=int, default=0, help="dry-run 시 자세별 상한")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_inputs: List[Dict] = []
    grid_axes_record = {
        "frames_axis": FRAMES_AXIS,
        "ratio_axis": RATIO_AXIS,
        "severity_axis": SEVERITY_AXIS,
        "warn_fail_split": WARN_FAIL_SPLIT,
    }
    pose_records = {}
    for pose in ALL_POSES:
        pose_target = pose.target_count
        # 임시로 target 줄임 (limit 옵션)
        if args.limit > 0:
            object.__setattr__(pose, "target_count", min(pose_target, args.limit))
        items = build_for_pose(pose, rng)
        if args.limit > 0:
            object.__setattr__(pose, "target_count", pose_target)
        all_inputs.extend(items)
        pose_records[pose.key] = {
            "name_en": pose.name_en,
            "name_kr": pose.name_kr,
            "has_scorer": pose.has_scorer,
            "phase_names": pose.phase_names,
            "feature_names": pose.feature_names or pose.fallback_features,
            "target_count": pose.target_count if args.limit == 0 else min(pose.target_count, args.limit),
            "actual_count": len(items),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in all_inputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    seed_record = {
        "seed": args.seed,
        "limit": args.limit,
        "instruction": SHARED_INSTRUCTION,
        "grid_axes": grid_axes_record,
        "poses": pose_records,
        "total_inputs": len(all_inputs),
    }
    args.seed_out.parent.mkdir(parents=True, exist_ok=True)
    with args.seed_out.open("w", encoding="utf-8") as f:
        json.dump(seed_record, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_inputs)} inputs → {args.out}")
    print(f"Wrote seed record → {args.seed_out}")
    for k, v in pose_records.items():
        print(f"  {k}: {v['actual_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
