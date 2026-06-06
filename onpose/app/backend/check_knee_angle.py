"""무릎 각도 측정 검증 스크립트 (v10dev).

합성 랜드마크(다리를 쭉 편 자세 / 90도 굽힌 자세)를 analyze_frame 에 넣어
무릎 각도가 올바르게(쭉 편 다리 ~180도) 측정되는지 확인한다.

사용법 (backend 폴더 안에서):
    .venv\\Scripts\\python.exe check_knee_angle.py

기대 결과:
    STRAIGHT LEG -> angle_src=2d_direct, knee≈180
    BENT KNEE    -> angle_src=2d_direct, knee≈90
리프터를 강제로 켜서 비교하려면:  set ONPOSE_USE_LIFTER=1  후 재실행.
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # cwd=backend 를 path 에 (app 패키지)
from app import ai_bridge  # noqa: E402


def make_landmarks(knee_straight=True):
    """33 x [x, y, z, vis]. 옆모습, 다리는 전방 수평.
    knee_straight=True 면 hip-knee-ankle 가 수평 일직선 → 무릎 ~180."""
    lm = [[0.5, 0.5, 0.0, 1.0] for _ in range(33)]

    def put(i, x, y, z=0.0, v=1.0):
        lm[i] = [x, y, z, v]

    put(0, 0.42, 0.40)                       # nose
    put(11, 0.45, 0.45); put(12, 0.45, 0.47)  # shoulders
    put(23, 0.40, 0.60); put(24, 0.40, 0.62)  # hips
    if knee_straight:
        put(25, 0.55, 0.60); put(27, 0.70, 0.60)  # L knee, ankle (수평 일직선)
        put(26, 0.55, 0.62); put(28, 0.70, 0.62)  # R knee, ankle
    else:
        put(25, 0.55, 0.60); put(27, 0.55, 0.45)  # 무릎 ~90 굽힘
        put(26, 0.55, 0.62); put(28, 0.55, 0.47)
    return lm


def run(label, knee_straight):
    lm = make_landmarks(knee_straight)
    state = {"exercise_id": "spine_stretch", "reps": 8}
    last = None
    for _ in range(90):                       # 리프터 윈도우(81) 충분히 채움
        last = ai_bridge.analyze_frame(lm, state)
    angles = {k: round(v, 1) for k, v in last["angles"].items()}
    msgs = [x["msg"] for x in last["feedback"]]
    print(f"[{label}] angle_src={state.get('angle_src')}  angles={angles}")
    for m in msgs:
        print(f"    feedback: {m}")


if __name__ == "__main__":
    sess = ai_bridge._try_load_lifter()
    print(f"lifter loaded={sess is not None}  USE_LIFTER_DEFAULT={ai_bridge._USE_LIFTER_DEFAULT}\n")
    run("STRAIGHT LEG (expect knee~180)", True)
    run("BENT KNEE   (expect knee~90)", False)
