"""
사용자가 취하고 있는 자세를 hip/knee/trunk 각도로 자동 분류.

KNN/룰베이스 — 학습된 모델 없이 정답 각도와의 거리로 가장 가까운 자세 추천.
사용자가 자세 박스를 선택하지 않아도 자세를 잡으면 자동 인식.

가중치는 angle_scorer 와 동일하게 critical/normal 따라가서
일관성 유지 (knee 0.45 / hip 0.40 / trunk 0.15).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .angle_scorer import RUBRICS, PoseRubric


def _weighted_distance(angles: Dict[str, float], rubric: PoseRubric) -> float:
    """현재 각도와 자세의 정답 사이의 가중 거리"""
    dist = 0.0
    wsum = 0.0
    for key, spec in rubric.angles.items():
        if key not in angles:
            continue
        diff = abs(float(angles[key]) - spec.target_deg) / max(spec.tolerance_deg, 1.0)
        dist += spec.weight * diff
        wsum += spec.weight
    return dist / max(wsum, 1e-8)


def classify_pose(angles: Dict[str, float], confidence_margin: float = 0.6
                  ) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Args:
        angles: {"hip": ..., "knee": ..., "trunk": ...}
        confidence_margin: 2등과 1등 거리 비가 이 값보다 작으면 "확신 부족"으로 처리

    Returns:
        (pose_key 또는 None, 신뢰도 0~1, {각 자세: 정규화 거리})
    """
    scores = {key: _weighted_distance(angles, rub) for key, rub in RUBRICS.items()}
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    if not ordered:
        return None, 0.0, scores
    best_key, best_dist = ordered[0]
    # 거리 → 신뢰도 환산: 거리 0 = 1.0, 거리 2.0 이상 = 0.0
    confidence = max(0.0, min(1.0, 1.0 - best_dist / 2.0))
    # 차상위와 너무 가깝거나 거리가 너무 크면 확신 부족
    if len(ordered) > 1:
        second_dist = ordered[1][1]
        if second_dist > 0:
            ratio = best_dist / second_dist
            if ratio > confidence_margin:
                confidence *= 0.5
    if best_dist > 1.5:
        return None, confidence, scores
    return best_key, confidence, scores


def auto_detect_with_smoothing(angle_history: List[Dict[str, float]], window: int = 15
                              ) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    최근 N프레임의 각도를 평균낸 뒤 분류 — 일시적 흔들림에 robust.
    """
    if not angle_history:
        return None, 0.0, {}
    recent = angle_history[-window:]
    if not recent:
        return None, 0.0, {}
    avg = {k: sum(d.get(k, 0.0) for d in recent) / len(recent)
           for k in recent[0].keys()}
    return classify_pose(avg)
