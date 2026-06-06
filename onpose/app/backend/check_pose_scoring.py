"""자세별 채점 곡선 검증 (카메라 불필요).

사용법 (backend 폴더 안에서):
    .venv\\Scripts\\python.exe check_pose_scoring.py

곡선 단위표 + 자세별 실시간 피드백(정답/나쁨) + 최종 리포트 + analyze_frame 엔드투엔드를
한 번에 확인한다. 정답값/곡선은 core/pose_targets.py 의 POSE_TARGETS 에서 정의.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from app import ai_bridge  # noqa: E402  (import 시 자세별 곡선 overlay 적용)
from core.distribution_scorer import score_frame_distribution, score_sequence_distribution  # noqa: E402
from core.pose_targets import POSE_TARGETS, _angle_subscore  # noqa: E402

print("=== 1) 곡선 단위표 (값->점수, exp=기대, XX=불일치>1) ===")
TABLE = {
    ("Spine_Stretch", "knee"): [(180, 100), (175, 100), (170, 90), (160, 70), (150, 50)],
    ("Spine_Stretch", "hip"): [(40, 100), (55, 100), (70, 80), (90, 62), (110, 44)],
    ("Bridging", "knee"): [(60, 100), (65, 100), (70, 90), (75, 80), (80, 70)],
    ("Bridging", "hip"): [(180, 100), (175, 100), (170, 90), (160, 70)],
    ("The_Seal", "knee"): [(30, 100), (35, 100), (40, 90), (45, 80), (50, 70)],
    ("The_Seal", "hip"): [(80, 100), (85, 100), (90, 90), (95, 80), (100, 70)],
}
for (pose, ang), pairs in TABLE.items():
    c = POSE_TARGETS[pose][ang]
    cells = []
    for v, exp in pairs:
        sub, _ = _angle_subscore(v, c)
        cells.append(f"{v}->{sub:.0f}({exp}){'' if abs(sub - exp) <= 1.0 else 'XX'}")
    print(f"  {pose:<14}/{ang:<4}: " + "  ".join(cells))

print("\n=== 2) 프레임 채점 + 실시간 피드백 ===")
def check(ex, angles, tag):
    rub, _pk = ai_bridge._rubric_for(ex)
    per = score_frame_distribution(angles, rub)
    fb = ai_bridge._make_feedback_items(angles, rub)
    print(f"  [{ex} {tag}] angles={angles}  score={per['score']}  채점각도={list(rub.angles.keys())}")
    for f in fb:
        print(f"        {f['level']}: {f['msg']}")

check("spine_stretch", {"knee": 178, "hip": 45}, "정답")
check("spine_stretch", {"knee": 150, "hip": 110}, "나쁨")
check("bridging", {"knee": 60, "hip": 178}, "정답")
check("bridging", {"knee": 90, "hip": 150}, "나쁨")
check("the_seal", {"knee": 30, "hip": 80}, "정답")
check("the_seal", {"knee": 60, "hip": 110}, "나쁨")

print("\n=== 3) 최종 리포트(시퀀스) — spine 나쁨 5프레임 ===")
rub, _ = ai_bridge._rubric_for("spine_stretch")
summ = score_sequence_distribution([{"knee": 150, "hip": 110}] * 5, rub)
print(f"  mean_score={summ['mean_score']}  per_angle_mean_pen={summ.get('per_angle_mean_pen')}")
