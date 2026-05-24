"""
각도 기반 rule-based 채점 모듈.

설계 원칙 (멘토링 피드백 반영):
- 사람마다 체형/골격이 다르기 때문에 중요한 각도에 가중치를 부여
- 정답 각도와의 거리(deg)에 비례한 점수 감점 (rule-based)
- O/X 정량 평가: 각도가 정답 ±tolerance 안에 들어온 프레임 비율
- 깊은 전문가적 요소(근육 활성도 등) 대신 정답 각도 설정만 사용
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AngleSpec:
    """단일 관절 각도의 정답 + 가중치 + 허용오차"""
    name: str               # 화면 표기용 한글 이름
    target_deg: float       # 정답 각도 (mean from training data)
    tolerance_deg: float    # ±tolerance 안이면 OK (O)
    weight: float           # 가중치 (자세에서 이 각도의 중요도)
    importance: str = "normal"  # 'critical' | 'normal' | 'minor' — 시각화/메시지용


@dataclass
class PoseRubric:
    """한 자세에 대한 채점 기준 — 여러 각도의 묶음"""
    pose_key: str
    pose_name_kr: str
    angles: Dict[str, AngleSpec] = field(default_factory=dict)
    pass_threshold: float = 80.0   # O/X 80% 이상이면 "합격"

    def total_weight(self) -> float:
        return sum(a.weight for a in self.angles.values())


# -----------------------------------------------------------------------------
# Rubrics — golden_standard JSON + 멘토링 피드백 기반
# -----------------------------------------------------------------------------
RUBRICS: Dict[str, PoseRubric] = {
    "The_Seal": PoseRubric(
        pose_key="The_Seal",
        pose_name_kr="더 씰",
        angles={
            # 핵심: 무릎(고관절 굴곡 자세에서 가장 critical), 가중치 최대
            "knee": AngleSpec("무릎 각도", target_deg=35.64, tolerance_deg=10.0, weight=0.45, importance="critical"),
            "hip": AngleSpec("고관절 각도", target_deg=80.73, tolerance_deg=12.0, weight=0.40, importance="critical"),
            "trunk": AngleSpec("상체 각도", target_deg=120.0, tolerance_deg=20.0, weight=0.15, importance="normal"),
        },
    ),
    "Bridging": PoseRubric(
        pose_key="Bridging",
        pose_name_kr="브릿징",
        angles={
            "hip": AngleSpec("고관절 신전", target_deg=170.0, tolerance_deg=15.0, weight=0.50, importance="critical"),
            "knee": AngleSpec("무릎 굴곡", target_deg=90.0, tolerance_deg=15.0, weight=0.35, importance="critical"),
            "trunk": AngleSpec("상체 정렬", target_deg=170.0, tolerance_deg=15.0, weight=0.15, importance="normal"),
        },
    ),
    "Spine_Stretch": PoseRubric(
        pose_key="Spine_Stretch",
        pose_name_kr="스파인 스트레치",
        angles={
            # 앉아서 상체를 굽히는 동작 — 무릎은 완전 신전, 고관절은 깊게 굴곡
            "hip": AngleSpec("고관절 굴곡", target_deg=80.0, tolerance_deg=15.0, weight=0.45, importance="critical"),
            "knee": AngleSpec("무릎 신전", target_deg=175.0, tolerance_deg=8.0, weight=0.35, importance="critical"),
            "trunk": AngleSpec("상체 굴곡", target_deg=140.0, tolerance_deg=20.0, weight=0.20, importance="normal"),
        },
    ),
}


def score_single_frame(angles: Dict[str, float], rubric: PoseRubric) -> Dict:
    """
    한 프레임의 각도들을 채점.

    Returns:
        {
            "score": 0~100,         # 가중 평균 점수
            "ox": True/False,       # 가중치 적용 통과 여부 (score >= pass_threshold)
            "details": {각도키: {"value": x, "target": y, "diff": d, "ok": bool, "subscore": 0~100}}
        }
    """
    details = {}
    weighted_sum = 0.0
    total_w = rubric.total_weight() or 1.0

    for key, spec in rubric.angles.items():
        value = float(angles.get(key, 0.0))
        diff = abs(value - spec.target_deg)
        # 0deg 오차 = 100점, tolerance 내에서는 선형 감점, 그 이후엔 가파른 감점
        if diff <= spec.tolerance_deg:
            sub = 100.0 - (diff / spec.tolerance_deg) * 30.0   # tolerance 안 → 70~100점
        else:
            extra = diff - spec.tolerance_deg
            sub = max(0.0, 70.0 - (extra / spec.tolerance_deg) * 70.0)
        ok = diff <= spec.tolerance_deg
        details[key] = {
            "value": round(value, 2),
            "target": spec.target_deg,
            "diff": round(diff, 2),
            "tolerance": spec.tolerance_deg,
            "ok": ok,
            "subscore": round(sub, 1),
            "weight": spec.weight,
            "importance": spec.importance,
            "name_kr": spec.name,
        }
        weighted_sum += sub * spec.weight

    score = weighted_sum / total_w
    return {
        "score": round(score, 1),
        "ox": score >= rubric.pass_threshold,
        "details": details,
    }


def score_sequence(angle_frames: Sequence[Dict[str, float]], rubric: PoseRubric) -> Dict:
    """
    프레임 시퀀스에 대해 채점한 뒤 집계.

    Returns:
        {
            "n_frames": int,
            "mean_score": 0~100,         # 평균 가중 점수
            "ox_accuracy": 0~100,         # O/X 정량 정확도 (% of frames PASS)
            "verdict": "합격" | "보통" | "부족",
            "per_angle_accuracy": {각도: 0~100},   # 각 각도가 tolerance에 들어온 비율
            "per_angle_mean_diff": {각도: deg},     # 평균 오차
            "top_issue": 각도키,            # 가장 점수 낮은 각도
            "top_issue_name_kr": str,
            "frame_scores": [점수…],         # 그래프용
        }
    """
    if not angle_frames:
        return {
            "n_frames": 0, "mean_score": 0.0, "ox_accuracy": 0.0,
            "verdict": "측정 실패", "per_angle_accuracy": {}, "per_angle_mean_diff": {},
            "top_issue": None, "top_issue_name_kr": "-", "frame_scores": [],
        }

    per_frame = [score_single_frame(a, rubric) for a in angle_frames]
    scores = [f["score"] for f in per_frame]
    ox_pass = sum(1 for f in per_frame if f["ox"])
    ox_accuracy = 100.0 * ox_pass / len(per_frame)
    mean_score = float(np.mean(scores))

    # 각도별 accuracy + mean diff
    per_angle_accuracy: Dict[str, float] = {}
    per_angle_mean_diff: Dict[str, float] = {}
    per_angle_mean_subscore: Dict[str, float] = {}
    for key in rubric.angles.keys():
        oks = [f["details"][key]["ok"] for f in per_frame]
        diffs = [f["details"][key]["diff"] for f in per_frame]
        subs = [f["details"][key]["subscore"] for f in per_frame]
        per_angle_accuracy[key] = round(100.0 * sum(oks) / len(oks), 1)
        per_angle_mean_diff[key] = round(float(np.mean(diffs)), 2)
        per_angle_mean_subscore[key] = round(float(np.mean(subs)), 1)

    # 가장 문제된 각도 = 평균 subscore 최저
    top_issue = min(per_angle_mean_subscore, key=per_angle_mean_subscore.get)
    top_issue_name_kr = rubric.angles[top_issue].name

    if mean_score >= 85.0:
        verdict = "훌륭해요"
    elif mean_score >= 70.0:
        verdict = "좋아요"
    elif mean_score >= 55.0:
        verdict = "조금 더"
    else:
        verdict = "다시 도전"

    return {
        "n_frames": len(per_frame),
        "mean_score": round(mean_score, 1),
        "ox_accuracy": round(ox_accuracy, 1),
        "verdict": verdict,
        "per_angle_accuracy": per_angle_accuracy,
        "per_angle_mean_diff": per_angle_mean_diff,
        "per_angle_mean_subscore": per_angle_mean_subscore,
        "top_issue": top_issue,
        "top_issue_name_kr": top_issue_name_kr,
        "frame_scores": [round(s, 1) for s in scores],
    }


def get_rubric(pose_key: str) -> Optional[PoseRubric]:
    return RUBRICS.get(pose_key)


def load_calibrated_rubrics(json_path) -> Dict[str, PoseRubric]:
    """reports/rubric_calibrated.json 을 읽어 PoseRubric 사전으로 반환.
    데이터셋에서 자동 산출된 정답 각도/허용오차를 적용.
    """
    import json
    from pathlib import Path
    p = Path(json_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for pose_key, val in data.items():
        angles = {}
        for ak, ang in val.get("angles", {}).items():
            angles[ak] = AngleSpec(
                name=ang["name_kr"],
                target_deg=float(ang["target_deg"]),
                tolerance_deg=float(ang["tolerance_deg"]),
                weight=float(ang["weight"]),
                importance=str(ang.get("importance", "normal")),
            )
        out[pose_key] = PoseRubric(
            pose_key=pose_key,
            pose_name_kr=val.get("name_kr", pose_key),
            angles=angles,
            pass_threshold=float(val.get("pass_threshold", 80.0)),
        )
    return out


def apply_calibrated_rubrics(json_path) -> bool:
    """현재 RUBRICS 전역을 calibrated 버전으로 교체"""
    cal = load_calibrated_rubrics(json_path)
    if not cal:
        return False
    RUBRICS.update(cal)
    return True
