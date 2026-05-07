"""자세별 v4 scorer 메타데이터 단일 출처.

`live_ai_coach_v4_bundle/pilates_temporal_lifter/runs/<run>/model.json` 의 phase_names·
feature_names 를 그대로 옮겨 적은 것이고, Spine Stretch 만 fallback 분기 (scorer 미학습)다.
합성 input 의 phase_counts·top_issue_counts 키가 런타임 분포와 정확히 일치하도록
이 파일에 정의된 이름만 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PoseMeta:
    key: str
    name_en: str
    name_kr: str
    has_scorer: bool
    phase_names: List[str]
    feature_names: List[str]
    fallback_features: List[str] = field(default_factory=list)
    target_count: int = 0


THE_SEAL = PoseMeta(
    key="The_Seal",
    name_en="The Seal",
    name_kr="더 씰",
    has_scorer=True,
    phase_names=["READY", "OUTBOUND", "APEX", "RETURN"],
    feature_names=["hip_angle", "knee_angle", "hip_velocity", "knee_velocity"],
    target_count=200,
)

BRIDGING = PoseMeta(
    key="Bridging",
    name_en="Bridging",
    name_kr="브릿징",
    has_scorer=True,
    phase_names=["START", "MIDDLE", "END"],
    feature_names=[
        "hip_angle",
        "knee_angle",
        "trunk_angle",
        "hip_symmetry",
        "knee_symmetry",
        "hip_velocity",
        "knee_velocity",
        "trunk_velocity",
    ],
    target_count=175,
)

SPINE_STRETCH = PoseMeta(
    key="Spine_Stretch",
    name_en="Spine Stretch",
    name_kr="스파인 스트레치",
    has_scorer=False,
    phase_names=[],
    feature_names=[],
    fallback_features=["hip", "knee"],
    target_count=125,
)

ALL_POSES: List[PoseMeta] = [THE_SEAL, BRIDGING, SPINE_STRETCH]
POSE_BY_KEY = {p.key: p for p in ALL_POSES}


SHARED_INSTRUCTION = (
    "당신은 정확하지만 격려하는 필라테스 강사입니다. 다음 분석 요약을 바탕으로 "
    "한국어 존댓말로 정확히 3문장 자세 교정 피드백을 작성하세요. "
    "(1) 전반적 품질, (2) top issue feature 또는 phase 기반 가장 중요한 교정점, "
    "(3) 다음 시도 시 적용할 단순 큐. 구현 디테일(Mahalanobis 등)은 언급하지 마세요."
)
