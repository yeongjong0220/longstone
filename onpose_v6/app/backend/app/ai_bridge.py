"""OnPose v6 통합 AI 브리지.

기존 mock을 onpose_v6/core 모듈로 교체:
  - 각도 계산: pose_pipeline.angles_from_landmarks_2d (visibility 가중)
  - 채점:     distribution_scorer.score_frame_distribution + score_sequence_distribution
  - 피드백:   feedback_engine.generate_feedback (Gemini 또는 offline 친근체 템플릿)
  - 자세 자동 인식: pose_classifier.classify_pose

frontend는 MediaPipe Tasks Vision으로 33개 landmark를 직접 추출해 WebSocket으로 보냄
([x, y, z, visibility] × 33). 백엔드는 그걸 SimpleNamespace로 wrap해 v6에 전달.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# onpose_v6 core 모듈 경로 — 이제 onpose_v6 안에 있으므로 자동 산출.
# 이 파일: onpose_v6/app/backend/app/ai_bridge.py
# parents: [0]=app(파일내), [1]=backend, [2]=app(상위), [3]=onpose_v6 (목표)
_THIS = Path(__file__).resolve()
_ONPOSE_DIR = _THIS.parents[3]
if not (_ONPOSE_DIR / "core" / "__init__.py").exists():
    # fallback: 다른 일반 후보 위치 탐색
    for cand in [_THIS.parents[4] / "onpose_v6", _THIS.parents[2] / "onpose_v6"]:
        if (cand / "core" / "__init__.py").exists():
            _ONPOSE_DIR = cand
            break
    else:
        raise RuntimeError(f"onpose_v6 core not found near {_THIS}")
sys.path.insert(0, str(_ONPOSE_DIR))
print(f"[ai_bridge] onpose_v6 root = {_ONPOSE_DIR}")

# Backend에서는 MediaPipe / mediapipe.tasks 가 필요 없음 (frontend가 landmarks 추출).
# pose_pipeline 모듈이 import 시점에 mediapipe 를 끌어오니 stub 주입.
import types as _types
for _mod_name in ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python",
                   "mediapipe.tasks.python.vision"):
    sys.modules.setdefault(_mod_name, _types.ModuleType(_mod_name))

# .env 로드 (Gemini API key 가능 시)
try:
    from dotenv import load_dotenv
    load_dotenv(_ONPOSE_DIR.parent / ".env")
except ImportError:
    pass
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# v6 코어
from core.angle_scorer import RUBRICS, apply_calibrated_rubrics, get_rubric
from core.distribution_scorer import (
    load_distribution_rubrics, score_frame_distribution, score_sequence_distribution,
)
from core.feedback_engine import generate_feedback
from core.pose_classifier import classify_pose
# pose_pipeline 안의 일부 함수만 직접 가져온다 (전체 모듈 import는 lifter 검색까지 함)
import importlib.util
_pp_spec = importlib.util.spec_from_file_location("pose_pipeline_lite",
    _ONPOSE_DIR / "core" / "pose_pipeline.py")
try:
    _pp = importlib.util.module_from_spec(_pp_spec)
    _pp_spec.loader.exec_module(_pp)
    angle_deg = _pp.angle_deg
    angles_from_landmarks_2d = _pp.angles_from_landmarks_2d
except Exception as _e:
    # pose_pipeline 로드 실패 → 순수 numpy 구현으로 fallback
    print(f"[ai_bridge] pose_pipeline load failed ({_e}), using inline angle calc")
    import numpy as _np

    def angle_deg(a, b, c):
        u = a - b; v = c - b
        u = u / max(float(_np.linalg.norm(u)), 1e-8)
        v = v / max(float(_np.linalg.norm(v)), 1e-8)
        return float(_np.degrees(_np.arccos(_np.clip(_np.dot(u, v), -1.0, 1.0))))

    def angles_from_landmarks_2d(landmarks, min_vis: float = 0.4):
        import numpy as _np
        if landmarks is None:
            return {"hip": 0.0, "knee": 0.0, "trunk": 0.0}
        def pt(i):
            lm = landmarks[i]
            return _np.array([lm.x, lm.y], dtype=_np.float32)
        def vis(i):
            return float(getattr(landmarks[i], "visibility", 1.0))
        def safe(a, b, c):
            v = min(vis(a), vis(b), vis(c))
            if v < min_vis:
                return None, 0.0
            return angle_deg(pt(a), pt(b), pt(c)), v
        lh, vlh = safe(11, 23, 25); rh, vrh = safe(12, 24, 26)
        lk, vlk = safe(23, 25, 27); rk, vrk = safe(24, 26, 28)
        def merge(a, va, b, vb):
            if a is None and b is None: return 0.0
            if a is None: return b
            if b is None: return a
            return (a * va + b * vb) / max(va + vb, 1e-8)
        hip = merge(lh, vlh, rh, vrh)
        knee = merge(lk, vlk, rk, vrk)
        hip_mid = (pt(23) + pt(24)) / 2.0
        sh_mid = (pt(11) + pt(12)) / 2.0
        trunk = angle_deg(hip_mid, sh_mid, pt(0))
        return {"hip": hip, "knee": knee, "trunk": trunk}

from core.pose_guide import POSE_HINTS

# Calibrated rubric 적용 (있으면)
_cal_path = _ONPOSE_DIR / "reports" / "rubric_calibrated.json"
if _cal_path.exists():
    apply_calibrated_rubrics(_cal_path)
    print(f"[ai_bridge] calibrated rubrics applied from {_cal_path}")

# Distribution-based rubrics 로드 (z-score + velocity)
_stats_path = _ONPOSE_DIR / "reports" / "pose_stats.json"
_distribution_rubrics: Dict[str, Any] = {}
if _stats_path.exists():
    _distribution_rubrics = load_distribution_rubrics(_stats_path)
    print(f"[ai_bridge] distribution rubrics loaded: {list(_distribution_rubrics.keys())}")

# ────────────────────────────────────────────────────────────────
# 2D → 3D Lifter (ONNX Runtime)
# 데스크탑 onpose_v6_coach.py 와 동일한 채점 파이프라인을 모바일 backend 에서 재현.
# import 안 되거나 모델 없으면 2D fallback (analyze_frame 안에서 자동 처리).
# ────────────────────────────────────────────────────────────────
import numpy as np  # noqa: E402

_LIFTER_JOINT_ORDER = [
    "Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip", "LKnee", "Rknee",
    "LAnkle", "RAnkle", "Hip",
]
_LIFTER_JOINT_IDX = {name: i for i, name in enumerate(_LIFTER_JOINT_ORDER)}

# MediaPipe 33 → Lifter 15 (mean of group when multiple sources)
_MP_TO_LIFTER = {
    "Head": [0], "Neck": [11, 12], "LShoulder": [11], "RShoulder": [12],
    "LElbow": [13], "RElbow": [14], "LWrist": [15], "RWrist": [16],
    "LHip": [23], "RHip": [24], "LKnee": [25], "Rknee": [26],
    "LAnkle": [27], "RAnkle": [28], "Hip": [23, 24],
}

_LIFTER_WINDOW = 81

# ONNX runtime / 모델은 lazy 하게 로드 (요청이 처음 들어올 때까지 미룸)
_lifter_session = None
_lifter_load_attempted = False
_lifter_load_err: Optional[str] = None


def _try_load_lifter():
    """ONNX Runtime InferenceSession 을 한 번만 로드. 실패하면 None 유지."""
    global _lifter_session, _lifter_load_attempted, _lifter_load_err
    if _lifter_load_attempted:
        return _lifter_session
    _lifter_load_attempted = True
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        _lifter_load_err = f"onnxruntime import 실패: {e}"
        print(f"[ai_bridge] lifter disabled — {_lifter_load_err}")
        return None
    onnx_path = _ONPOSE_DIR / "reports" / "lifter_causal_int8.onnx"
    if not onnx_path.exists():
        onnx_path = _ONPOSE_DIR / "reports" / "lifter_causal.onnx"
    if not onnx_path.exists():
        _lifter_load_err = f"ONNX 모델 없음: {onnx_path}"
        print(f"[ai_bridge] lifter disabled — {_lifter_load_err}")
        return None
    try:
        _lifter_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        print(f"[ai_bridge] lifter loaded from {onnx_path.name}  "
              f"(inputs={[(i.name, i.shape) for i in _lifter_session.get_inputs()]})")
    except Exception as e:
        _lifter_load_err = f"ONNX 로드 실패: {e}"
        print(f"[ai_bridge] lifter disabled — {_lifter_load_err}")
        _lifter_session = None
    return _lifter_session


def _mp_landmarks_to_lifter_2d(landmarks, min_visibility: float = 0.35) -> np.ndarray:
    """MediaPipe 33 landmark(SimpleNamespace 리스트) → Lifter 15-joint 2D 좌표."""
    frame = np.zeros((len(_LIFTER_JOINT_ORDER), 2), dtype=np.float32)
    if landmarks is None:
        return frame
    for out_idx, name in enumerate(_LIFTER_JOINT_ORDER):
        pts = []
        for src in _MP_TO_LIFTER[name]:
            lm = landmarks[src]
            if float(getattr(lm, "visibility", 1.0)) < min_visibility:
                continue
            pts.append([float(lm.x), float(lm.y)])
        if pts:
            frame[out_idx] = np.asarray(pts, dtype=np.float32).mean(axis=0)
    return frame


def _normalize_skeleton(coords: np.ndarray) -> np.ndarray:
    """dataset.normalize_skeleton inline 포팅 (root=Hip, scale=median(shoulder/torso/hip width))."""
    arr = np.nan_to_num(coords.astype(np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    idx = _LIFTER_JOINT_IDX
    root = arr[:, idx["Hip"]:idx["Hip"] + 1, :]
    arr = arr - root
    shoulder_width = np.linalg.norm(arr[:, idx["LShoulder"], :] - arr[:, idx["RShoulder"], :], axis=-1)
    torso = np.linalg.norm(arr[:, idx["Neck"], :] - arr[:, idx["Hip"], :], axis=-1)
    hip_width = np.linalg.norm(arr[:, idx["LHip"], :] - arr[:, idx["RHip"], :], axis=-1)
    shoulder_width = np.nan_to_num(shoulder_width, nan=0.0, posinf=0.0, neginf=0.0)
    torso = np.nan_to_num(torso, nan=0.0, posinf=0.0, neginf=0.0)
    hip_width = np.nan_to_num(hip_width, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.nanmedian(np.maximum.reduce([shoulder_width, torso, hip_width])))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    arr /= scale
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _build_observation_mask(x2d: np.ndarray) -> np.ndarray:
    x2d = np.asarray(x2d, dtype=np.float32)
    mask = np.ones((*x2d.shape[:2], 1), dtype=np.float32)
    finite = np.isfinite(x2d).all(axis=-1, keepdims=True)
    near_zero = (np.abs(np.nan_to_num(x2d, nan=0.0)).sum(axis=-1, keepdims=True) < 1e-8)
    mask[~finite] = 0.0
    mask[near_zero] = 0.0
    return mask


def _lifter_predict_latest(frames_2d_buf: List[np.ndarray]) -> Optional[np.ndarray]:
    """최근 frames_2d (15x2 리스트) → 마지막 프레임 3D (15x3)."""
    sess = _try_load_lifter()
    if sess is None or not frames_2d_buf:
        return None
    arr = np.asarray(frames_2d_buf, dtype=np.float32)         # (T, 15, 2)
    arr = _normalize_skeleton(arr)
    obs = _build_observation_mask(arr)                         # (T, 15, 1)
    x = np.concatenate([arr, obs], axis=-1)                    # (T, 15, 3)
    # pad to window_size by repeating first frame
    T = x.shape[0]
    if T < _LIFTER_WINDOW:
        pad = np.repeat(x[:1], _LIFTER_WINDOW - T, axis=0)
        x = np.concatenate([pad, x], axis=0)
    elif T > _LIFTER_WINDOW:
        x = x[-_LIFTER_WINDOW:]
    inp = x[None, ...].astype(np.float32)                      # (1, W, 15, 3)
    try:
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
    except Exception as e:
        print(f"[ai_bridge] lifter inference 실패: {e}")
        return None
    pose3d = out[0]                                            # (1, W, 15, 3)
    return pose3d[0, -1].astype(np.float32)


def _angle_deg_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _angles_from_3d_visibility_weighted(frame3d: np.ndarray, landmarks, min_vis: float = 0.35) -> Dict[str, float]:
    """pose_pipeline.lifted_angles_visibility_weighted 의 backend inline 포팅."""
    idx = _LIFTER_JOINT_IDX
    lhip = _angle_deg_np(frame3d[idx["Neck"]],  frame3d[idx["LHip"]], frame3d[idx["LKnee"]])
    rhip = _angle_deg_np(frame3d[idx["Neck"]],  frame3d[idx["RHip"]], frame3d[idx["Rknee"]])
    lknee = _angle_deg_np(frame3d[idx["LHip"]], frame3d[idx["LKnee"]], frame3d[idx["LAnkle"]])
    rknee = _angle_deg_np(frame3d[idx["RHip"]], frame3d[idx["Rknee"]], frame3d[idx["RAnkle"]])
    trunk = _angle_deg_np(frame3d[idx["Hip"]],  frame3d[idx["Neck"]], frame3d[idx["Head"]])
    if landmarks is None:
        return {"hip": (lhip + rhip) / 2.0, "knee": (lknee + rknee) / 2.0, "trunk": trunk}
    def _vmin(*ids: int) -> float:
        return min(float(getattr(landmarks[i], "visibility", 1.0)) for i in ids)
    vlh = _vmin(11, 23, 25)
    vrh = _vmin(12, 24, 26)
    vlk = _vmin(23, 25, 27)
    vrk = _vmin(24, 26, 28)
    if vlh < min_vis: vlh = 0.0
    if vrh < min_vis: vrh = 0.0
    if vlk < min_vis: vlk = 0.0
    if vrk < min_vis: vrk = 0.0
    hip = (lhip * vlh + rhip * vrh) / (vlh + vrh) if (vlh + vrh) > 1e-6 else (lhip + rhip) / 2.0
    knee = (lknee * vlk + rknee * vrk) / (vlk + vrk) if (vlk + vrk) > 1e-6 else (lknee + rknee) / 2.0
    return {"hip": hip, "knee": knee, "trunk": trunk}

# Exercise ID (frontend) ↔ Pose key (v6) 매핑
_EXERCISE_TO_POSE = {
    "the_seal": "The_Seal",
    "spine_stretch": "Spine_Stretch",
    "bridging": "Bridging",
}

_PHASES = ["ready", "entry", "core", "return"]


def _landmarks_to_objs(landmarks: List[List[float]]) -> List[SimpleNamespace]:
    """[[x,y,z,vis], ...] → SimpleNamespace 리스트 (MediaPipe-like)"""
    out = []
    for lm in landmarks:
        if len(lm) >= 4:
            x, y, z, v = lm[0], lm[1], lm[2], lm[3]
        elif len(lm) == 3:
            x, y, z, v = lm[0], lm[1], lm[2], 1.0
        else:
            x, y, z, v = 0.0, 0.0, 0.0, 0.0
        out.append(SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=float(v)))
    return out


def _rubric_for(exercise_id: str):
    """exercise_id → DistributionRubric (있으면) 또는 PoseRubric"""
    pose_key = _EXERCISE_TO_POSE.get(exercise_id, exercise_id)
    if pose_key in _distribution_rubrics:
        return _distribution_rubrics[pose_key], pose_key
    rub = get_rubric(pose_key)
    return rub, pose_key


def _detect_phase(angles: Dict[str, float], rubric, state: dict) -> str:
    """단순 phase 추정 — 정답 각도와의 거리 + 시간으로 ready/entry/core/return.

    state에 마지막 score를 저장해 변화 감지.
    """
    if rubric is None:
        return "core"
    per = score_frame_distribution(angles, rubric)
    score = per.get("score", 0)
    last_score = state.get("last_score", 0)
    state["last_score"] = score
    # 점수가 낮으면 ready/entry, 높아지면 core, 다시 낮아지면 return
    if score < 40:
        return "ready" if last_score < 40 else "entry"
    if score >= 75:
        return "core"
    if last_score >= 75 and score < 75:
        return "return"
    return "entry"


def _detect_rep_count(angles: Dict[str, float], rubric, state: dict) -> int:
    """간단 반복 카운터 — 점수가 임계값 위/아래로 전환될 때 count++"""
    THRESH = 70.0
    per = score_frame_distribution(angles, rubric) if rubric else {"score": 0}
    score = per.get("score", 0)
    was_high = state.get("rep_was_high", False)
    is_high = score >= THRESH
    if is_high and not was_high:
        state["rep_count"] = state.get("rep_count", 0) + 1
    state["rep_was_high"] = is_high
    return state.get("rep_count", 0)


def _make_feedback_items(angles: Dict[str, float], rubric) -> List[Dict[str, str]]:
    """프레임 단위 피드백 (ok/warn/err 아이콘 + 짧은 메시지)"""
    items: List[Dict[str, str]] = []
    if rubric is None:
        return [{"level": "ok", "msg": "측정 중입니다"}]
    per = score_frame_distribution(angles, rubric)
    issues = []
    for k, spec in rubric.angles.items():
        d = per.get("details", {}).get(k, {})
        if not d.get("ok", True):
            v = d.get("value", 0); t = spec.target_deg; diff = d.get("diff", abs(v - t))
            direction = "더 굽혀 주세요" if v < t else "더 펴 주세요"
            issues.append({"level": "warn",
                           "msg": f"{spec.name} {diff:.0f}° 차이 — {direction}"})
    if not issues:
        items.append({"level": "ok", "msg": "핵심 각도 모두 정답 범위 안에 있어요"})
        hint = POSE_HINTS.get(rubric.pose_key if hasattr(rubric, "pose_key") else "", "")
        if hint:
            items.append({"level": "ok", "msg": hint})
    else:
        items.extend(issues[:2])
    return items


def analyze_frame(landmarks: list[list[float]], state: dict[str, Any]) -> dict[str, Any]:
    """매 frame 채점.

    Returns CoachingFrame dict:
      t, phase, rep_count, set_count, angles, score, status, feedback[]
    """
    t = state.get("t", 0.0)
    exercise_id = state.get("exercise_id", "")
    rubric, pose_key = _rubric_for(exercise_id)

    # frame_idx 카운트
    state["frame_idx"] = state.get("frame_idx", 0) + 1

    # 빈 landmarks 케이스
    if not landmarks or len(landmarks) < 29:
        return {
            "t": t, "phase": "ready", "rep_count": state.get("rep_count", 0),
            "set_count": state.get("set_count", 0),
            "angles": {"hip": 0.0, "knee": 0.0, "trunk": 0.0},
            "score": 0, "status": "err",
            "feedback": [{"level": "err", "msg": "전신이 보이지 않아요. 카메라에서 떨어져 주세요."}],
        }

    lms = _landmarks_to_objs(landmarks)

    # ── 2D → 3D Lifter (가능하면 데스크탑과 동일 경로) ─────────────
    frame3d: Optional[np.ndarray] = None
    angle_src = "2d_fallback"
    try:
        sess = _try_load_lifter()
        if sess is not None:
            buf: List[np.ndarray] = state.setdefault("lifter_buf", [])
            buf.append(_mp_landmarks_to_lifter_2d(lms))
            if len(buf) > _LIFTER_WINDOW:
                del buf[: len(buf) - _LIFTER_WINDOW]
            frame3d = _lifter_predict_latest(buf)
    except Exception as _e:
        # 어떤 이유로든 lifter 경로 실패 → 2D fallback (조용히)
        frame3d = None

    if frame3d is not None:
        angles = _angles_from_3d_visibility_weighted(frame3d, lms)
        angle_src = "3d_lifted"
    else:
        angles = angles_from_landmarks_2d(lms)
    state["angle_src"] = angle_src
    # angle_history 누적 (세션 종료 시 종합 채점에 사용)
    state.setdefault("angle_history", []).append(angles)

    # 채점
    if rubric is None:
        per = {"score": 75, "ox": False, "details": {}}
    else:
        per = score_frame_distribution(angles, rubric)

    phase = _detect_phase(angles, rubric, state)
    rep_count = _detect_rep_count(angles, rubric, state)
    reps_per_set = state.get("reps", 8)
    set_count = rep_count // max(1, reps_per_set)

    score = int(round(per.get("score", 0)))
    if score >= 80:
        status = "good"
    elif score >= 60:
        status = "warn"
    else:
        status = "err"

    feedback_items = _make_feedback_items(angles, rubric)

    return {
        "t": t,
        "phase": phase,
        "rep_count": rep_count,
        "set_count": set_count,
        "angles": {k: float(v) for k, v in angles.items()},
        "score": score,
        "status": status,
        "feedback": feedback_items,
    }


def generate_coaching(session_summary: dict[str, Any]) -> dict[str, Any]:
    """세션 종료 시 종합 피드백 (친근체 LLM).

    session_summary expected keys:
      exercise_id, frame_count, angle_history (list[dict])
    """
    exercise_id = session_summary.get("exercise_id", "")
    rubric, pose_key = _rubric_for(exercise_id)
    angle_history = session_summary.get("angle_history") or []

    if rubric is None or not angle_history:
        return {
            "score_avg": 0,
            "good_points": [],
            "improvements": ["측정된 프레임이 부족했어요. 다시 시도해 주세요."],
            "llm_msg": "조금만 더 가까이서 다시 시도해 봐요!",
        }

    summary = score_sequence_distribution(angle_history, rubric)
    pose_kr = getattr(rubric, "pose_name_kr", pose_key)
    feedback = generate_feedback(pose_kr, summary,
                                  api_key=GOOGLE_API_KEY,
                                  prefer_online=bool(GOOGLE_API_KEY))

    # good/improvement 분리
    good = []
    improvements = []
    per_acc = summary.get("per_angle_accuracy", {})
    for k, spec in rubric.angles.items():
        acc = per_acc.get(k, 0)
        if acc >= 80:
            good.append(f"{spec.name}이(가) 정답 범위 안에 잘 들어왔어요 ({acc:.0f}%)")
        else:
            diff = summary.get("per_angle_mean_diff", {}).get(k, 0)
            improvements.append(
                f"{spec.name}을(를) 평균 {abs(diff):.0f}° 더 조절해 보세요"
            )

    return {
        "score_avg": int(round(summary.get("mean_score", 0))),
        "good_points": good[:3],
        "improvements": improvements[:3],
        "llm_msg": feedback.get("text", ""),
    }
