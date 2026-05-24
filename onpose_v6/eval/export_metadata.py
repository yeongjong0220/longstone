"""
모바일/임베디드 이식을 위한 자세 채점 메타데이터 JSON 내보내기.

이 한 파일이 있으면 모바일 앱은 angle_scorer.py를 다시 구현하지 않고도
같은 룰베이스 채점/O-X 판정/친근체 템플릿을 그대로 사용할 수 있다.

출력: reports/onpose_metadata.json
스키마:
{
  "version": "1.0",
  "poses": {
    "The_Seal": {
        "name_kr": "더 씰",
        "pass_threshold": 80.0,
        "angles": {
            "knee": {
                "name_kr": "무릎 각도",
                "target_deg": 35.64,
                "tolerance_deg": 10.0,
                "weight": 0.45,
                "importance": "critical"
            }, ...
        },
        "tips": [...],
        "hint": "...",
        "safety": "..."
    }, ...
  },
  "feedback_templates": {...},
  "guide_skeleton_2d": {...}
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from core.angle_scorer import RUBRICS
from core.feedback_engine import _PRAISE_LINES, _PRAISE_HIGH, _CUE_BY_ISSUE, _END_LINES
from core.pose_guide import GUIDE_SKELETON_2D, POSE_HINTS, POSE_SAFETY, POSE_TIPS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] /
                                                    "reports" / "onpose_metadata.json")
    args = ap.parse_args()

    poses = {}
    for key, rub in RUBRICS.items():
        poses[key] = {
            "name_kr": rub.pose_name_kr,
            "pass_threshold": rub.pass_threshold,
            "angles": {
                ak: {
                    "name_kr": spec.name,
                    "target_deg": spec.target_deg,
                    "tolerance_deg": spec.tolerance_deg,
                    "weight": spec.weight,
                    "importance": spec.importance,
                }
                for ak, spec in rub.angles.items()
            },
            "tips": POSE_TIPS.get(key, []),
            "hint": POSE_HINTS.get(key, ""),
            "safety": POSE_SAFETY.get(key, ""),
            "guide_skeleton_2d": GUIDE_SKELETON_2D.get(key, {}),
        }

    metadata = {
        "version": "1.0",
        "schema_url": "https://github.com/your-repo/onpose-mobile-schema",
        "poses": poses,
        "feedback_templates": {
            "praise": _PRAISE_LINES,
            "praise_high": _PRAISE_HIGH,
            "cues_by_issue": _CUE_BY_ISSUE,
            "endings": _END_LINES,
        },
        "verdicts": {
            "ranges": [
                {"min": 85, "label": "훌륭해요"},
                {"min": 70, "label": "좋아요"},
                {"min": 55, "label": "조금 더"},
                {"min": 0,  "label": "다시 도전"},
            ],
        },
        "joint_order": [
            "Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
            "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle", "Hip",
        ],
        "mediapipe_to_lifter_map": {
            "Head": [0], "Neck": [11, 12], "LShoulder": [11], "RShoulder": [12],
            "LElbow": [13], "RElbow": [14], "LWrist": [15], "RWrist": [16],
            "LHip": [23], "RHip": [24], "LKnee": [25], "Rknee": [26],
            "LAnkle": [27], "RAnkle": [28], "Hip": [23, 24],
        },
        "scoring_formula": {
            "subscore_within_tol":  "100 - (diff_deg / tolerance_deg) * 30",
            "subscore_outside_tol": "max(0, 70 - (extra_deg / tolerance_deg) * 70)",
            "frame_score":          "sum(subscore * weight) / sum(weight)",
            "ox_pass": "frame_score >= pass_threshold",
            "session_ox_accuracy_pct": "100 * count(ox=True) / total_frames",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] exported metadata -> {args.out}")
    print(f"  poses: {list(poses.keys())}")
    print(f"  feedback templates: {sum(len(v) if isinstance(v, list) else len(v) for v in metadata['feedback_templates'].values())} entries")
    print()
    print("→ 모바일 앱은 이 JSON 한 파일만 번들하면 자세 채점 / O-X / 친근체를 그대로 사용 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
