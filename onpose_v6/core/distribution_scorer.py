"""
전문가 자세의 각도+각속도 *분포*를 기준으로 사용자 동작 채점.

기존 angle_scorer.py 의 단순 거리 기준 대신:
  - 각도   : z-score = (angle - mean) / std  → likelihood 변환
  - 각속도 : 시퀀스 frame-to-frame 변화량의 분포와 비교 (너무 빠름/굳어있음 페널티)

점수 계산:
  - per-frame: 각 각도의 z-score → 부드러운 likelihood score (0~100)
  - per-session: 평균 + 각속도 분포 일치도 결합
  - 가중치는 angle_scorer의 critical/normal 그대로 따름

장점:
  - 표준편차가 큰 trunk 같은 각도는 자연스럽게 관대해짐
  - 표준편차가 작은 knee는 자연스럽게 엄격해짐
  - 사람마다 체형 다른 부분이 std에 반영되어 무난한 점수
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .angle_scorer import AngleSpec, PoseRubric, get_rubric


@dataclass
class VelocityDistribution:
    mean_dps: float          # degree per frame
    std_dps: float
    p75_dps: float
    p95_dps: float


@dataclass
class DistributionRubric:
    """PoseRubric 호환 + 추가 통계 (std, velocity) 보유.
    UI 코드는 .angles[k].name / .target_deg / .tolerance_deg / .weight / .importance 그대로 사용 가능.
    """
    pose_key: str
    pose_name_kr: str
    angles: Dict[str, AngleSpec] = field(default_factory=dict)   # PoseRubric 호환
    angle_std: Dict[str, float] = field(default_factory=dict)    # z-score 용
    velocity: Dict[str, VelocityDistribution] = field(default_factory=dict)
    pass_threshold: float = 75.0
    n_actors: int = 0

    def total_weight(self) -> float:
        return sum(a.weight for a in self.angles.values())


def _gaussian_score(z: float) -> float:
    """z-score → 점수 (0~100), 부드러운 곡선.

    |z|=0   → 100
    |z|=1   → ~88
    |z|=2   → ~65 (분포 내 95% 안)
    |z|=3   → ~40
    |z|=4+  → <25
    """
    z = abs(float(z))
    # tanh 기반 — 부드러운 감점
    return float(100.0 * math.exp(-(z ** 2) / 4.5))


def load_distribution_rubrics(stats_path: Path | str) -> Dict[str, DistributionRubric]:
    """reports/pose_stats.json + RUBRICS 의 weight/importance 결합.
    PoseRubric 호환 인터페이스: angles[k]는 AngleSpec, target_deg=mean, tolerance_deg=std*1.5.
    추가로 angle_std와 velocity 보유.
    """
    p = Path(stats_path)
    if not p.exists():
        return {}
    stats = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, DistributionRubric] = {}
    for pose_key, data in stats.items():
        rub = get_rubric(pose_key)
        if rub is None:
            continue
        angles_spec: Dict[str, AngleSpec] = {}
        angle_std: Dict[str, float] = {}
        for k, sd in data.get("stats", {}).items():
            if k not in rub.angles:
                continue
            spec = rub.angles[k]
            std = max(3.0, float(sd["std"]))
            angles_spec[k] = AngleSpec(
                name=spec.name,
                target_deg=float(sd["median"]),     # 중앙값을 정답으로
                tolerance_deg=std * 1.5,            # 1.5σ를 허용오차로
                weight=spec.weight,
                importance=spec.importance,
            )
            angle_std[k] = std
        vel_dists = {}
        for k, vd in data.get("velocity_stats", {}).items():
            vel_dists[k] = VelocityDistribution(
                mean_dps=float(vd["mean_dps"]),
                std_dps=max(0.5, float(vd["std_dps"])),
                p75_dps=float(vd["p75_dps"]),
                p95_dps=float(vd["p95_dps"]),
            )
        out[pose_key] = DistributionRubric(
            pose_key=pose_key,
            pose_name_kr=rub.pose_name_kr,
            angles=angles_spec,
            angle_std=angle_std,
            velocity=vel_dists,
            pass_threshold=rub.pass_threshold,
            n_actors=int(data.get("n_actors", 0)),
        )
    return out


def score_frame_distribution(angles: Dict[str, float], rubric: DistributionRubric) -> Dict:
    """한 프레임 — z-score 기반 likelihood 점수.
    UI 호환: angle_scorer.score_single_frame과 동일한 dict 구조 반환.
    """
    details = {}
    weighted_sum = 0.0
    total_w = 0.0
    for k, spec in rubric.angles.items():
        v = float(angles.get(k, 0.0))
        mean = spec.target_deg
        std = rubric.angle_std.get(k, 5.0)
        z = (v - mean) / std
        sub = _gaussian_score(z)
        diff = abs(v - mean)
        ok = diff <= spec.tolerance_deg
        details[k] = {
            "value": round(v, 2),
            "target": mean,
            "diff": round(diff, 2),
            "tolerance": spec.tolerance_deg,
            "z_score": round(float(z), 3),
            "subscore": round(sub, 1),
            "ok": bool(ok),
            "weight": spec.weight,
            "importance": spec.importance,
            "name_kr": spec.name,
        }
        weighted_sum += sub * spec.weight
        total_w += spec.weight
    score = weighted_sum / max(total_w, 1e-8)
    return {
        "score": round(score, 1),
        "ox": score >= rubric.pass_threshold,
        "details": details,
    }


def _velocity_likelihood(user_vel: np.ndarray, dist: VelocityDistribution) -> float:
    """사용자 각속도 분포 vs 전문가 분포 → 0~100 점수.

    - 자세 유지(정지)는 OK → 사용자 mean이 expert mean보다 낮아도 80점 이상
    - 너무 빠르게 (전문가 p95 초과) 흔들면 가파른 감점
    - 비슷한 속도면 만점
    """
    if len(user_vel) == 0 or dist.mean_dps <= 0:
        return 85.0
    user_mean = float(np.mean(user_vel))
    expert_mean = dist.mean_dps
    expert_p95 = dist.p95_dps
    # 1) 사용자가 전문가보다 느림 (자세 유지) — 관대하게
    if user_mean <= expert_mean:
        # 정지(0)도 80점 보장, mean에 도달하면 100
        return float(80.0 + 20.0 * (user_mean / max(expert_mean, 1e-6)))
    # 2) p95 안쪽 (자연스러운 변동) → 100 → 80 선형
    if user_mean <= expert_p95:
        extra = (user_mean - expert_mean) / max(expert_p95 - expert_mean, 1e-6)
        return float(100.0 - 20.0 * extra)
    # 3) p95 초과 (너무 빠름) → 가파른 감점
    over = (user_mean - expert_p95) / max(expert_p95, 1e-6)
    return float(max(20.0, 80.0 - 60.0 * over))


def score_sequence_distribution(angle_frames: Sequence[Dict[str, float]],
                                 rubric: DistributionRubric) -> Dict:
    """시퀀스 채점 — 프레임별 z-score 점수 + 각속도 likelihood 결합"""
    if not angle_frames:
        return {
            "n_frames": 0, "mean_score": 0.0, "ox_accuracy": 0.0,
            "verdict": "측정 실패", "per_angle_accuracy": {},
            "per_angle_mean_diff": {}, "per_angle_mean_z": {},
            "velocity_score": 0.0, "velocity_user": {}, "velocity_expert": {},
            "top_issue": None, "top_issue_name_kr": "-", "frame_scores": [],
        }
    per_frame = [score_frame_distribution(a, rubric) for a in angle_frames]
    scores = [f["score"] for f in per_frame]
    ox_pass = sum(1 for f in per_frame if f["ox"])
    ox_acc = 100.0 * ox_pass / len(per_frame)
    mean_score = float(np.mean(scores))

    # 각도별 통계 (각 frame 의 z-score 분포)
    per_angle_acc: Dict[str, float] = {}
    per_angle_diff: Dict[str, float] = {}
    per_angle_mean_z: Dict[str, float] = {}
    per_angle_mean_sub: Dict[str, float] = {}
    for k in rubric.angles.keys():
        oks = [f["details"][k]["ok"] for f in per_frame]
        zs = [f["details"][k]["z_score"] for f in per_frame]
        subs = [f["details"][k]["subscore"] for f in per_frame]
        per_angle_acc[k] = round(100.0 * sum(oks) / len(oks), 1)
        per_angle_mean_z[k] = round(float(np.mean(np.abs(zs))), 3)
        per_angle_mean_sub[k] = round(float(np.mean(subs)), 1)
        vals = [float(angle_frames[i].get(k, 0.0)) for i in range(len(angle_frames))]
        per_angle_diff[k] = round(float(np.mean(vals)) - rubric.angles[k].target_deg, 2)

    # 각속도 채점
    velocity_score = 0.0
    velocity_user: Dict[str, float] = {}
    velocity_expert: Dict[str, float] = {}
    vw_sum = 0.0
    for k, vdist in rubric.velocity.items():
        if k not in rubric.angles:
            continue
        vals = np.asarray([float(angle_frames[i].get(k, 0.0)) for i in range(len(angle_frames))])
        if len(vals) < 2:
            continue
        user_vel = np.abs(np.diff(vals))
        user_mean = float(np.mean(user_vel))
        sub = _velocity_likelihood(user_vel, vdist)
        velocity_user[k] = round(user_mean, 3)
        velocity_expert[k] = round(vdist.mean_dps, 3)
        w = rubric.angles[k].weight
        velocity_score += sub * w
        vw_sum += w
    velocity_score = velocity_score / vw_sum if vw_sum > 0 else 70.0

    # 최종 점수: 각도 80% + 각속도 20% (각도가 자세의 1차 지표)
    final_score = mean_score * 0.8 + velocity_score * 0.2

    # 가장 문제된 각도
    top_issue = min(per_angle_mean_sub, key=per_angle_mean_sub.get)
    top_issue_name_kr = rubric.angles[top_issue].name

    if final_score >= 85: verdict = "훌륭해요"
    elif final_score >= 70: verdict = "좋아요"
    elif final_score >= 55: verdict = "조금 더"
    else: verdict = "다시 도전"

    return {
        "n_frames": len(per_frame),
        "mean_score": round(float(final_score), 1),
        "angle_score": round(mean_score, 1),
        "velocity_score": round(velocity_score, 1),
        "ox_accuracy": round(ox_acc, 1),
        "verdict": verdict,
        "per_angle_accuracy": per_angle_acc,
        "per_angle_mean_diff": per_angle_diff,
        "per_angle_mean_z": per_angle_mean_z,
        "per_angle_mean_subscore": per_angle_mean_sub,
        "velocity_user_dps": velocity_user,
        "velocity_expert_dps": velocity_expert,
        "top_issue": top_issue,
        "top_issue_name_kr": top_issue_name_kr,
        "frame_scores": [round(s, 1) for s in scores],
    }


def diagnostic_print(rubric: DistributionRubric) -> str:
    """rubric 진단 출력 — 어떤 분포를 쓰는지 확인용"""
    lines = [f"  Pose: {rubric.pose_name_kr}  (n_actors={rubric.n_actors})"]
    for k, spec in rubric.angles.items():
        v = rubric.velocity.get(k)
        std = rubric.angle_std.get(k, 0.0)
        vstr = f"v_mean {v.mean_dps:.2f}°/f" if v else "(no velocity)"
        lines.append(
            f"    {k:<6}  μ={spec.target_deg:>6.2f}°  σ={std:>5.2f}°  "
            f"tol={spec.tolerance_deg:>5.2f}°  w={spec.weight:.2f}  {vstr}"
        )
    return "\n".join(lines)
