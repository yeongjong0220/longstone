"""
자세별 시각/텍스트 가이드.

- POSE_TIPS  : 선택 화면 / 인트로 화면에 보여줄 한국어 팁
- POSE_HINTS : 측정 중 사용자가 보고 따라할 짧은 큐
- GUIDE_SKELETON_2D: 정답 자세를 카메라 영상 위에 반투명 골격으로 오버레이
                     (좌표는 카메라 화면 비율 기준 0~1, 자동으로 픽셀 변환)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

POSE_TIPS: Dict[str, List[str]] = {
    "The_Seal": [
        "옆모습이 카메라에 잘 보이도록 서거나 앉아 주세요",
        "무릎을 가슴쪽으로 접고, 양손은 발목 안쪽을 가볍게 잡아요",
        "허리 둥글림을 유지하면서 살짝 흔들 듯이 호흡해요",
        "★ 핵심 각도: 무릎 ≈ 36°, 고관절 ≈ 81°",
    ],
    "Spine_Stretch": [
        "다리를 곧게 펴고 앉아 카메라에 옆모습 보여 주세요",
        "팔을 앞으로 나란히 뻗고, 척추를 길게 만든다는 느낌",
        "숨을 내쉬며 천천히 상체를 앞으로 굽혀요",
        "★ 핵심 각도: 무릎 ≈ 175°(거의 펴짐), 고관절 ≈ 80°",
    ],
    "Bridging": [
        "바닥에 등을 대고 누워, 카메라에 옆/위 시점이 잡히도록 해주세요",
        "발은 어깨 너비, 무릎은 90° 정도로 세워요",
        "엉덩이를 천천히 들어 올려 무릎-어깨가 일직선이 되도록",
        "★ 핵심 각도: 고관절 ≈ 170°(신전), 무릎 ≈ 90°",
    ],
}

POSE_HINTS: Dict[str, str] = {
    "The_Seal": "무릎은 접고, 허리는 둥글게!",
    "Spine_Stretch": "다리는 곧게, 상체는 앞으로!",
    "Bridging": "엉덩이를 천천히 들어 올리세요!",
}

# 자세별 안전/주의 메모 (멘토링: "안으로 굽었는데 밖으로 굽었다 같은 표현 자제")
POSE_SAFETY: Dict[str, str] = {
    "The_Seal": "허리에 통증 있으면 무리하지 마세요.",
    "Spine_Stretch": "무릎이 아프면 살짝 굽혀도 괜찮아요.",
    "Bridging": "목에 힘 빼고, 견갑골로 바닥을 지지하세요.",
}

# 정답 자세 가이드 골격 (카메라 좌표 0~1 normalized)
# 매우 단순화된 정답 자세 실루엣 — 사용자가 카메라 위에 자기 자세를 맞추는 데만 사용
GUIDE_SKELETON_2D: Dict[str, Dict] = {
    "The_Seal": {
        "joints": {
            "head": (0.50, 0.40),
            "neck": (0.50, 0.48),
            "shoulder": (0.52, 0.50),
            "hip": (0.50, 0.60),
            "knee": (0.45, 0.50),
            "ankle": (0.42, 0.46),
            "hand": (0.45, 0.48),
        },
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hand"), ("hand", "ankle")],
    },
    "Spine_Stretch": {
        "joints": {
            "head": (0.45, 0.45),
            "neck": (0.47, 0.50),
            "shoulder": (0.49, 0.52),
            "hip": (0.55, 0.65),
            "knee": (0.70, 0.65),
            "ankle": (0.85, 0.65),
            "hand": (0.30, 0.55),
        },
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hand")],
    },
    "Bridging": {
        "joints": {
            "head": (0.20, 0.70),
            "neck": (0.25, 0.65),
            "shoulder": (0.30, 0.62),
            "hip": (0.50, 0.50),
            "knee": (0.65, 0.60),
            "ankle": (0.70, 0.75),
            "hand": (0.20, 0.72),
        },
        "bones": [("head", "neck"), ("neck", "shoulder"), ("shoulder", "hip"),
                  ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hand")],
    },
}
