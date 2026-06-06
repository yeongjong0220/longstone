"""실시간 무릎 임계값(170) + 최종 잘한점/개선점 규칙 기반 검증 (카메라 불필요).

실행: backend/ 에서  .venv/Scripts/python.exe check_feedback_rules.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())   # app 패키지 import 가능하게

from app import ai_bridge          # 먼저 로드 → onpose_v8 root 를 sys.path 에 추가
from core.pose_targets import (
    _fb_spine_knee, _fb_spine_hip, angle_report, _angle_subscore, POSE_TARGETS,
)


def section(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ── 1. 실시간 무릎 임계값: 170° 까지 ok ──────────────────────────────
section("1. 실시간 무릎 _fb_spine_knee (170° 경계)")
cases = [190, 185, 180, 175, 172, 170, 169, 160, 135, 134, 100]
for v in cases:
    st, msg = _fb_spine_knee(v)
    print(f"  knee={v:>3}°  → {st:<4}  {msg}")
expect = {190: "warn", 180: "ok", 170: "ok", 169: "warn", 134: "err"}
ok = all(_fb_spine_knee(v)[0] == e for v, e in expect.items())
print(f"  [무릎 경계 기대치 일치] {ok}")

section("1b. 실시간 고관절 _fb_spine_hip (75° 경계)")
for v in [40, 55, 75, 76, 90, 110, 119, 120, 130]:
    st, msg = _fb_spine_hip(v)
    print(f"  hip={v:>3}°  → {st:<4}  {msg}")
hip_exp = {55: "ok", 75: "ok", 76: "warn", 119: "warn", 120: "err"}
hip_ok = all(_fb_spine_hip(v)[0] == e for v, e in hip_exp.items())
print(f"  [고관절 경계 기대치 일치] {hip_ok}")

section("1c. 고관절 채점 곡선 _angle_subscore (≤75 만점 plateau)")
hip_curve = POSE_TARGETS["Spine_Stretch"]["hip"]
for v in [40, 60, 75, 90, 110, 120]:
    sub, info = _angle_subscore(v, hip_curve)
    print(f"  hip={v:>3}°  → score={sub:>5.1f}  ok={info['ok']}  dir={info['direction']}")
plateau_ok = (_angle_subscore(75, hip_curve)[0] == 100.0
              and _angle_subscore(40, hip_curve)[0] == 100.0
              and _angle_subscore(90, hip_curve)[0] < 100.0)
print(f"  [≤75 만점 / >75 감점] {plateau_ok}")


# ── 2. 최종 리포트 문구: 수치 없음 ──────────────────────────────────
section("2. angle_report 문구 (각도 수치 미언급 확인)")
for k, name in [("hip", "고관절"), ("knee", "무릎")]:
    g = angle_report("Spine_Stretch", k, True, name)
    b = angle_report("Spine_Stretch", k, False, name)
    print(f"  {k} 잘한점 : {g}")
    print(f"  {k} 개선점 : {b}")
no_digit = all(not any(c.isdigit() for c in s)
               for k, name in [("hip", "고관절"), ("knee", "무릎")]
               for s in (angle_report("Spine_Stretch", k, True, name),
                         angle_report("Spine_Stretch", k, False, name)))
print(f"  [숫자 미포함] {no_digit}")


# ── 3. generate_coaching: 너그러운 잘한점 (초록 30%↑ → 잘한점) ────────
section("3. generate_coaching 시나리오 (Spine_Stretch)")

def hist(knee_vals, hip_vals):
    return [{"knee": k, "hip": h, "trunk": 30.0} for k, h in zip(knee_vals, hip_vals)]

def run(label, knee_vals, hip_vals):
    summary = {
        "exercise_id": "spine_stretch",
        "frame_count": len(knee_vals),
        "angle_history": hist(knee_vals, hip_vals),
    }
    rep = ai_bridge.generate_coaching(summary)
    print(f"\n  [{label}]  score_avg={rep['score_avg']}")
    print(f"    잘한점 : {rep['good_points']}")
    print(f"    개선점 : {rep['improvements']}")
    print(f"    코치   : {rep['llm_msg'][:50]}...")
    return rep

# A) 무릎은 줄곧 곧음(초록 多), 고관절은 절반만 잘 숙임 → 둘 다 잘한점 기대
run("무릎 곧음 + 고관절 절반 숙임",
    [178]*10 + [176]*10,
    [50]*10 + [115]*10)

# B) 무릎 곧음(초록), 고관절은 거의 안 숙임 → 무릎=잘한점, 고관절=개선점("더 숙여")
run("무릎 곧음 + 고관절 안 숙임",
    [178]*20,
    [120]*20)

# C) 무릎 굽음(초록 0), 고관절 잘 숙임 → 무릎=개선점("더 펴"), 고관절=잘한점
run("무릎 굽음 + 고관절 잘 숙임",
    [120]*20,
    [45]*20)

# D) 무릎 172(170~185 초록), 30%만 초록인 고관절 → 무릎 잘한점, 고관절 잘한점(30%↑)
run("무릎 172 + 고관절 30% 초록",
    [172]*20,
    [50]*6 + [120]*14)

print("\n검증 끝.")
