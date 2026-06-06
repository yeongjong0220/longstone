"""자세별(3종) 정답 각도 + 채점 곡선 — 측정-정답 단일 출처.

측정(각도 계산)은 그대로 두고, 자세마다 다른 '정답 자세'를 여기서 정의한다.
실시간 피드백과 최종 결과 채점이 모두 이 곡선을 공유한다(distribution_scorer 한 곳에서 호출).

곡선 3종:
  - peak(target, band, slope)   : |v-target| ≤ band → 100, 초과분 1°당 slope점 감점 ("X에 가까울수록")
  - at_most(target, slope, ...) : v ≤ target → 100, 초과분 감점 (스파인 엉덩이: ≤55 만점)
  - at_least(target, slope)     : v ≥ target → 100, 미달분 감점 (180을 향하는 각도)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Curve:
    kind: str                       # "peak" | "at_most" | "at_least"
    target: float                   # peak 중심 / at_most·at_least 의 만점 경계
    band: float = 0.0               # peak: ±band 까지 만점
    slope: float = 2.0              # 벌점구간 1°당 감점
    floor: float = 0.0              # 최저 subscore (clamp)
    knee2: Optional[float] = None   # (옵션) 벌점거리 knee2 초과부터 slope2 적용 (2단 곡선)
    slope2: Optional[float] = None


# ── 사용자 확정 정답 (무릎+엉덩이만, 상체 제외) ─────────────────────────────
# "가까울수록" 곡선은 ±5° 만점(엄격). 무릎:엉덩이 = 5:5.
POSE_TARGETS: Dict[str, Dict[str, Curve]] = {
    "Spine_Stretch": {
        # 무릎: 180°에 가까울수록 (≥175 만점, 미달분 감점)
        "knee": Curve(kind="at_least", target=175.0, slope=2.0),
        # 엉덩이(어깨-엉덩이-무릎): ≤75° 만점, 초과 시 점차 감점 (앵커 75→100,90→80,110→62,120→53)
        "hip": Curve(kind="at_most", target=75.0, slope=1.33, knee2=15.0, slope2=0.91),
    },
    "Bridging": {
        "knee": Curve(kind="peak", target=60.0, band=5.0, slope=2.0),
        "hip": Curve(kind="at_least", target=175.0, slope=2.0),   # 180 지향
    },
    "The_Seal": {
        "knee": Curve(kind="peak", target=30.0, band=5.0, slope=2.0),
        "hip": Curve(kind="peak", target=80.0, band=5.0, slope=2.0),
    },
}

_ANGLE_WEIGHT = 0.5   # 무릎:엉덩이 = 5:5


def _angle_subscore(value: float, c: Curve) -> Tuple[float, Dict]:
    """value(측정 각도) + Curve → (0~100 subscore, info).

    info = {ideal, diff(벌점구간까지 거리), ok(만점구간 여부), direction("더 펴/더 굽혀"/None)}.
    피드백/리포트는 info 를 그대로 사용한다.
    """
    v = float(value)
    # 미검출/무효 랜드마크(≈0)는 만점으로 오인하지 않도록 0점 처리
    if v <= 1e-3:
        return 0.0, {"ideal": round(c.target, 1), "diff": 0.0, "ok": False, "direction": None}

    if c.kind == "peak":
        d = abs(v - c.target)
        pen = max(0.0, d - c.band)
        ok = d <= c.band
        direction = None if ok else ("더 펴 주세요" if v < c.target else "더 굽혀 주세요")
    elif c.kind == "at_most":
        pen = max(0.0, v - c.target)        # 정답보다 더 펴짐(큰 각) → 벌점
        ok = pen <= 0.0
        direction = None if ok else "더 굽혀 주세요"
    else:  # at_least
        pen = max(0.0, c.target - v)        # 정답보다 더 굽혀짐(작은 각) → 벌점
        ok = pen <= 0.0
        direction = None if ok else "더 펴 주세요"

    if c.knee2 is not None and c.slope2 is not None and pen > c.knee2:
        sub = 100.0 - c.slope * c.knee2 - c.slope2 * (pen - c.knee2)
    else:
        sub = 100.0 - c.slope * pen
    sub = max(c.floor, min(100.0, sub))
    return sub, {"ideal": round(c.target, 1), "diff": round(pen, 1),
                 "ok": bool(ok), "direction": direction}


def apply_targets(rubric) -> bool:
    """룹릭(PoseRubric/DistributionRubric)에 자세별 정답 곡선을 덮어쓴다.

    - 각 AngleSpec 에 .curve 부착 + target_deg/tolerance_deg/weight 정렬(레거시 호환).
    - 설정에 없는 각도(상체 등)는 rubric.angles 에서 제거 → 채점/피드백에서 제외.
    적용되면 True.
    """
    pose_key = getattr(rubric, "pose_key", "")
    cfg = POSE_TARGETS.get(pose_key)
    if not cfg:
        return False
    for k, curve in cfg.items():
        spec = rubric.angles.get(k)
        if spec is None:
            continue
        spec.curve = curve
        spec.target_deg = curve.target          # 레거시 reader 호환
        spec.tolerance_deg = curve.band         # peak band (plateau 는 0)
        spec.weight = _ANGLE_WEIGHT
    # 설정에 없는 각도 제거 (상체 trunk 등)
    for k in list(rubric.angles.keys()):
        if k not in cfg:
            del rubric.angles[k]
    return True


# ── 자세별 실시간 피드백 규칙 (status, message) ─────────────────────────────
# status: "ok"(초록) | "warn"(노랑) | "err"(빨강) — 아이콘/숫자/스켈레톤 원 색을 결정.
# Spine Stretch 는 시연 기준이라 명시 규칙. 그 외 자세는 curve 기반 일반 규칙.

def _fb_spine_hip(v: float) -> Tuple[str, str]:
    # 정답(plateau) 75° 이하 = 만점. 클수록(덜 숙임) 나쁨.
    if v <= 75.0:
        return "ok", "잘하고 있어요. 고관절 각도를 유지해주세요"
    if v < 120.0:
        return "warn", "허리를 앞으로 더 숙여주세요"
    return "err", "허리를 앞으로 쭉 숙여주세요"


def _fb_spine_knee(v: float) -> Tuple[str, str]:
    # 정답 180°(일자). 170°까지 잘한 자세(초록), 169°↓는 더 펴라고, 45°↑ 차이면 빨강.
    if v > 185.0:                        # 과신전(정답보다 큼)
        return "warn", "무릎을 일자로 펴주세요"
    if v >= 170.0:                       # 170~185 → 잘한 자세
        return "ok", "잘하고 있어요. 무릎 각도를 유지해주세요"
    if v >= 135.0:                       # 135~169 (11~45° 더 굽음)
        return "warn", "무릎을 더 곧게 펴주세요"
    return "err", "무릎이 너무 구부러져있어요. 일자로 펴주세요"   # v < 135 (45°↑ 더 굽음)


_POSE_FEEDBACK = {
    "Spine_Stretch": {"hip": _fb_spine_hip, "knee": _fb_spine_knee},
}


def angle_feedback(pose_key: str, angle_key: str, value: float, curve, name: str) -> Tuple[str, str]:
    """관절 1개의 (status, 메시지). pose별 명시 규칙 우선, 없으면 curve 기반 일반 규칙."""
    fn = _POSE_FEEDBACK.get(pose_key, {}).get(angle_key)
    if fn is not None:
        return fn(float(value))
    # 일반 규칙: curve 로 ok/warn/err + 방향 메시지 (다른 자세용)
    if curve is None:
        return "ok", f"{name} 측정 중"
    sub, info = _angle_subscore(value, curve)
    if info["ok"]:
        return "ok", f"잘하고 있어요. {name} 유지"
    status = "err" if sub < 45.0 else "warn"
    direction = info["direction"] or "조정해 주세요"
    return status, f"{name} {info['diff']:.0f}° 차이 — {direction}"


# ── 최종 리포트 잘한점/개선점 문구 (각도 수치 미언급, 실시간 톤) ───────────────
# 실시간 피드백과 같은 결의 문장이되, 숫자(몇도)는 빼고 행동 큐만 준다.
# 분류(잘한점/개선점)는 ai_bridge 가 실시간과 동일한 angle_feedback 으로 판정한다.
_SPINE_REPORT_GOOD = {
    "hip": "고관절 각도는 좋았어요",
    "knee": "무릎 각도는 좋았어요",
}
_SPINE_REPORT_IMPROVE = {
    "hip": "허리를 앞으로 더 숙여주세요",
    "knee": "무릎을 더 곧게 펴주세요",
}


def angle_report(pose_key: str, angle_key: str, is_good: bool, name: str,
                 direction: Optional[str] = None) -> str:
    """최종 리포트용 잘한점/개선점 한 줄. 각도 수치 미언급, 실시간 피드백 톤."""
    if pose_key == "Spine_Stretch":
        table = _SPINE_REPORT_GOOD if is_good else _SPINE_REPORT_IMPROVE
        if angle_key in table:
            return table[angle_key]
    # 일반 폴백 (Spine 외 자세)
    if is_good:
        return f"{name} 각도는 좋았어요"
    if direction:                       # "더 펴 주세요" / "더 굽혀 주세요"
        return f"{name}을(를) {direction}"
    return f"{name} 자세를 조금 더 신경 써 주세요"
