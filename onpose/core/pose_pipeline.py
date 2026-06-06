"""
MediaPipe → (옵션) TemporalLifter → 각도 추출 까지의 파이프라인.
v5_quality에서 분리/정리하여 v6 UI가 깔끔하게 호출만 하도록 함.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# pilates_temporal_lifter 가 두 군데에 있을 수 있음 — 가까운 쪽 우선
# __file__ = .../longstone/onpose_v6/core/pose_pipeline.py
#   parents[0] = core
#   parents[1] = onpose_v6
#   parents[2] = longstone   <- 여기에 pilates_temporal_lifter 가 있음
#   parents[3] = 인공지능캡스톤디자인
_THIS = Path(__file__).resolve()
_CANDIDATES = [
    _THIS.parents[2] / "pilates_temporal_lifter",                      # longstone/pilates_temporal_lifter
    _THIS.parents[2] / "live_ai_coach_v4_bundle" / "pilates_temporal_lifter",
    _THIS.parents[2] / "longstone_wonpark" / "pilates_temporal_lifter",
    _THIS.parents[3] / "pilates_temporal_lifter",
]
LIFTER_DIR = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
sys.path.insert(0, str(LIFTER_DIR))
print(f"[init] lifter dir: {LIFTER_DIR}  exists={LIFTER_DIR.exists()}")

try:
    from dataset import JOINT_ORDER                       # noqa: E402
    from runtime_lifting import OnlineTemporalLifter      # noqa: E402
except ImportError as e:
    raise RuntimeError(f"pilates_temporal_lifter를 찾을 수 없습니다: {e}\nLIFTER_DIR={LIFTER_DIR}")

JOINT_IDX = {name: idx for idx, name in enumerate(JOINT_ORDER)}


_DETECTOR_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def find_pose_landmarker(project_root: Path, variant: str = "heavy") -> Path:
    """pose_landmarker_{variant}.task 찾기, 없으면 다운로드.
    variant: 'lite'(가벼움, 핸드폰), 'full'(중간), 'heavy'(정확)
    """
    filename = f"pose_landmarker_{variant}.task"
    candidates = [
        project_root / filename,
        project_root.parent / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = _DETECTOR_URLS[variant]
    print(f"[init] downloading {filename} -> {target}")
    urllib.request.urlretrieve(url, target)
    return target


def build_detector(model_path: Path, lite_mode: bool = False):
    with open(model_path, "rb") as f:
        model_buffer = f.read()
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_buffer),
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.4 if lite_mode else 0.3,
            min_tracking_confidence=0.4 if lite_mode else 0.3,
        )
    )


def build_lifter(prefer: str = "causal") -> Optional[OnlineTemporalLifter]:
    runs = LIFTER_DIR / "runs"
    order = [
        "the_seal_progress3_angle_causal_v1",
        "the_seal_progress3_lift_only_v1",
    ] if prefer == "causal" else [
        "the_seal_progress3_lift_only_v1",
        "the_seal_progress3_angle_causal_v1",
    ]
    for name in order:
        p = runs / name / "best.pt"
        if p.exists():
            print(f"[init] loaded lifter checkpoint: {p}")
            return OnlineTemporalLifter(p, window_size=81)
    print("[init] no lifter checkpoint found → 2D-only mode")
    return None


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def lifted_angles(frame3d: np.ndarray | None) -> Dict[str, float]:
    """3D lifter 출력에서 hip/knee/trunk 각도 계산"""
    if frame3d is None:
        return {"hip": 0.0, "knee": 0.0, "trunk": 0.0}
    lhip = angle_deg(frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]])
    rhip = angle_deg(frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]])
    lknee = angle_deg(frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]], frame3d[JOINT_IDX["LAnkle"]])
    rknee = angle_deg(frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]], frame3d[JOINT_IDX["RAnkle"]])
    trunk = angle_deg(frame3d[JOINT_IDX["Hip"]], frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["Head"]])
    return {"hip": (lhip + rhip) / 2.0, "knee": (lknee + rknee) / 2.0, "trunk": trunk}


def angles_from_landmarks_2d(landmarks, min_vis: float = 0.4) -> Dict[str, float]:
    """MediaPipe 33점에서 직접 2D 각도 추정 (lifter 없을 때 fallback).

    옆모습/가려짐 대응:
      - 각 관절의 visibility를 가중치로 사용해 좌/우 가중 평균
      - 한 쪽이 visibility < min_vis 면 그쪽은 무시하고 반대쪽만 사용
    """
    if landmarks is None:
        return {"hip": 0.0, "knee": 0.0, "trunk": 0.0}

    def pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)

    def vis(idx: int) -> float:
        return float(getattr(landmarks[idx], "visibility", 1.0))

    # 좌/우 각도 + 그 각도가 신뢰할 만한지(=관여 관절 모두 visible)
    def safe_angle(a_i, b_i, c_i):
        v = min(vis(a_i), vis(b_i), vis(c_i))
        if v < min_vis:
            return None, 0.0
        return angle_deg(pt(a_i), pt(b_i), pt(c_i)), v

    sh_mid = (pt(11) + pt(12)) / 2.0   # 척추 상단(양 어깨 중점) = spine centerline top

    # 고관절 = 척추(어깨 중점)-엉덩이-무릎. 양 어깨 중점을 써 몸통 회전(3/4뷰)에 강건.
    def hip_safe(hip_i, knee_i):
        v = min(vis(11), vis(12), vis(hip_i), vis(knee_i))
        if v < min_vis:
            return None, 0.0
        return angle_deg(sh_mid, pt(hip_i), pt(knee_i)), v

    lhip, vlh = hip_safe(23, 25)
    rhip, vrh = hip_safe(24, 26)
    lknee, vlk = safe_angle(23, 25, 27)
    rknee, vrk = safe_angle(24, 26, 28)

    def merge(a, va, b, vb):
        if a is None and b is None:
            return 0.0
        if a is None: return b
        if b is None: return a
        # visibility 가중 평균
        return (a * va + b * vb) / max(va + vb, 1e-8)

    hip = merge(lhip, vlh, rhip, vrh)
    knee = merge(lknee, vlk, rknee, vrk)

    # 상체: 골반(평균) - 어깨(평균) - 코
    hip_mid = (pt(23) + pt(24)) / 2.0
    trunk = angle_deg(hip_mid, sh_mid, pt(0))
    return {"hip": hip, "knee": knee, "trunk": trunk}


def lifted_angles_visibility_weighted(frame3d: np.ndarray | None,
                                       landmarks=None,
                                       min_vis: float = 0.35) -> Dict[str, float]:
    """3D 출력을 쓰되, MediaPipe visibility로 좌/우 가중 평균.
    한쪽이 안 보이면 보이는 쪽만 사용해 옆모습 robustness 향상.
    """
    if frame3d is None:
        return {"hip": 0.0, "knee": 0.0, "trunk": 0.0}

    lhip = angle_deg(frame3d[JOINT_IDX["Neck"]],  frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]])
    rhip = angle_deg(frame3d[JOINT_IDX["Neck"]],  frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]])
    lknee = angle_deg(frame3d[JOINT_IDX["LHip"]], frame3d[JOINT_IDX["LKnee"]], frame3d[JOINT_IDX["LAnkle"]])
    rknee = angle_deg(frame3d[JOINT_IDX["RHip"]], frame3d[JOINT_IDX["Rknee"]], frame3d[JOINT_IDX["RAnkle"]])
    trunk = angle_deg(frame3d[JOINT_IDX["Hip"]],  frame3d[JOINT_IDX["Neck"]], frame3d[JOINT_IDX["Head"]])

    if landmarks is not None:
        # MediaPipe visibility로 가중
        vlh = min(float(getattr(landmarks[11], "visibility", 1.0)),
                  float(getattr(landmarks[23], "visibility", 1.0)),
                  float(getattr(landmarks[25], "visibility", 1.0)))
        vrh = min(float(getattr(landmarks[12], "visibility", 1.0)),
                  float(getattr(landmarks[24], "visibility", 1.0)),
                  float(getattr(landmarks[26], "visibility", 1.0)))
        vlk = min(float(getattr(landmarks[23], "visibility", 1.0)),
                  float(getattr(landmarks[25], "visibility", 1.0)),
                  float(getattr(landmarks[27], "visibility", 1.0)))
        vrk = min(float(getattr(landmarks[24], "visibility", 1.0)),
                  float(getattr(landmarks[26], "visibility", 1.0)),
                  float(getattr(landmarks[28], "visibility", 1.0)))
        # 임계값 이하는 0
        if vlh < min_vis: vlh = 0.0
        if vrh < min_vis: vrh = 0.0
        if vlk < min_vis: vlk = 0.0
        if vrk < min_vis: vrk = 0.0
        if vlh + vrh > 1e-6:
            hip = (lhip * vlh + rhip * vrh) / (vlh + vrh)
        else:
            hip = (lhip + rhip) / 2.0
        if vlk + vrk > 1e-6:
            knee = (lknee * vlk + rknee * vrk) / (vlk + vrk)
        else:
            knee = (lknee + rknee) / 2.0
        return {"hip": hip, "knee": knee, "trunk": trunk}
    return {"hip": (lhip + rhip) / 2.0, "knee": (lknee + rknee) / 2.0, "trunk": trunk}


class AngleSmoother:
    """각도 EMA 스무더 + outlier 클립.

    트레이닝 바지/헐렁한 옷으로 인한 jitter 완화:
      - EMA(alpha=0.4)로 부드럽게
      - 변화량 > step_limit 이면 outlier로 보고 step_limit 만큼만 이동
    """
    def __init__(self, alpha: float = 0.45, step_limit_deg: float = 12.0):
        self.alpha = alpha
        self.step_limit = step_limit_deg
        self.state: Dict[str, float] = {}

    def reset(self):
        self.state.clear()

    def __call__(self, angles: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in angles.items():
            v = float(v)
            if v <= 0.0:
                out[k] = self.state.get(k, 0.0)
                continue
            if k not in self.state:
                self.state[k] = v
            prev = self.state[k]
            delta = v - prev
            # outlier 큰 도약 제한
            if abs(delta) > self.step_limit:
                delta = float(np.sign(delta)) * self.step_limit
            # EMA
            new_val = prev + self.alpha * delta
            self.state[k] = new_val
            out[k] = new_val
        return out


class PoseAnglePipeline:
    """
    한 프레임 단위로 (이미지 → MediaPipe → [옵션] Lifter → [옵션] 후처리 → 각도) 처리.
    각 단계 latency를 외부 profiler에 보고할 수 있도록 hook 제공.

    enable_bone_lock=True 일 때 매 프레임 출력 3D에 bone-length 비율 보정을 적용
    (시간이 갈수록 누적된 시퀀스 기준으로 anchor 길이 추정 → 부드럽게 안정화).
    """

    def __init__(self, detector, lifter: Optional[OnlineTemporalLifter] = None, profiler=None,
                 enable_bone_lock: bool = True, enable_smoothing: bool = True,
                 enable_landmark_smooth: bool = True, enable_frame3d_smooth: bool = True,
                 enable_occlusion_blend: bool = True) -> None:
        self.detector = detector
        self.lifter = lifter
        self.profiler = profiler
        self.enable_bone_lock = enable_bone_lock
        self.enable_smoothing = enable_smoothing
        self.enable_landmark_smooth = enable_landmark_smooth
        self.enable_frame3d_smooth = enable_frame3d_smooth
        self.enable_occlusion_blend = enable_occlusion_blend
        self._recent_3d: list = []
        self._max_recent = 64
        self.smoother = AngleSmoother(alpha=0.45, step_limit_deg=12.0)
        # 새 안정화기들
        from core.occlusion_robust import (Frame3DSmoother, Landmark2DSmoother,
                                           OcclusionAwareJointBlender)
        self.landmark_smoother = Landmark2DSmoother(alpha=0.55, vis_threshold=0.35, hold_max_frames=15)
        self.frame3d_smoother = Frame3DSmoother(window=7, polyorder=2)
        self.occ_blender = OcclusionAwareJointBlender()

    def reset_lifter(self) -> None:
        if self.lifter is not None:
            self.lifter.reset()
        self._recent_3d = []
        self.smoother.reset()
        self.landmark_smoother.reset()
        self.frame3d_smoother.reset()
        self.occ_blender.reset()

    def _apply_bone_lock(self, frame3d):
        """온라인 bone-length lock — 최근 누적 시퀀스 기준 비율로 자식 관절 재구성"""
        from core.lifting_postprocess import enforce_bone_length
        self._recent_3d.append(frame3d)
        if len(self._recent_3d) > self._max_recent:
            self._recent_3d = self._recent_3d[-self._max_recent:]
        if len(self._recent_3d) < 4:
            return frame3d
        import numpy as np
        seq = np.asarray(self._recent_3d, dtype=np.float32)
        locked = enforce_bone_length(seq)
        return locked[-1]

    def process_frame(self, bgr_frame: np.ndarray) -> Tuple:
        """
        Returns:
            image_bgr (annotated 가능, 여기서는 원본 그대로 반환),
            landmarks (mediapipe pose_landmarks[0] 또는 None),
            angles (dict),
            frame3d (numpy or None),
            meta (dict with timings)
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.profiler:
            with self.profiler.measure("mediapipe"):
                results = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        else:
            results = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        # 1) 2D landmark 시간적 in-paint + EMA (헐렁한 옷/가려짐 jitter 완화)
        if landmarks is not None and self.enable_landmark_smooth:
            if self.profiler:
                with self.profiler.measure("landmark_smooth"):
                    landmarks = self.landmark_smoother(landmarks)
            else:
                landmarks = self.landmark_smoother(landmarks)

        frame3d = None
        if self.lifter is not None and landmarks is not None:
            if self.profiler:
                with self.profiler.measure("lifter"):
                    frame3d, _ = self.lifter.append_landmarks(landmarks)
            else:
                frame3d, _ = self.lifter.append_landmarks(landmarks)

            # 2) 3D 출력에 짧은 Savgol 스무딩 (5~7프레임)
            if frame3d is not None and self.enable_frame3d_smooth:
                if self.profiler:
                    with self.profiler.measure("frame3d_smooth"):
                        frame3d = self.frame3d_smoother(frame3d)
                else:
                    frame3d = self.frame3d_smoother(frame3d)

            # 3) Occlusion-aware joint blending (lifter 출력 vs 직전 프레임)
            if frame3d is not None and self.enable_occlusion_blend and landmarks is not None:
                # MediaPipe 33→lifter 15 visibility 매핑
                mp_to_lifter = {
                    "Head": 0, "Neck": (11, 12), "LShoulder": 11, "RShoulder": 12,
                    "LElbow": 13, "RElbow": 14, "LWrist": 15, "RWrist": 16,
                    "LHip": 23, "RHip": 24, "LKnee": 25, "Rknee": 26,
                    "LAnkle": 27, "RAnkle": 28, "Hip": (23, 24),
                }
                vis_15 = []
                for j_name in ["Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow",
                               "LWrist", "RWrist", "LHip", "RHip", "LKnee", "Rknee",
                               "LAnkle", "RAnkle", "Hip"]:
                    src = mp_to_lifter[j_name]
                    if isinstance(src, tuple):
                        vis_15.append(min(float(getattr(landmarks[s], "visibility", 1.0)) for s in src))
                    else:
                        vis_15.append(float(getattr(landmarks[src], "visibility", 1.0)))
                if self.profiler:
                    with self.profiler.measure("occlusion_blend"):
                        frame3d = self.occ_blender(frame3d, vis_15)
                else:
                    frame3d = self.occ_blender(frame3d, vis_15)

            # 4) Bone-length lock (마지막 — 길이 일관성 확보)
            if frame3d is not None and self.enable_bone_lock:
                if self.profiler:
                    with self.profiler.measure("bone_lock"):
                        frame3d = self._apply_bone_lock(frame3d)
                else:
                    frame3d = self._apply_bone_lock(frame3d)

        if frame3d is not None:
            # visibility 가중 + lifter 출력 사용 (옆모습 robustness)
            angles = lifted_angles_visibility_weighted(frame3d, landmarks)
            angle_src = "3d_lifted_locked" if self.enable_bone_lock else "3d_lifted"
        elif landmarks is not None:
            angles = angles_from_landmarks_2d(landmarks)
            angle_src = "2d_fallback"
        else:
            angles = {"hip": 0.0, "knee": 0.0, "trunk": 0.0}
            angle_src = "none"

        # 시간적 smoothing (트레이닝 바지/옷 흔들림 대응)
        if self.enable_smoothing:
            angles = self.smoother(angles)

        return bgr_frame, landmarks, angles, frame3d, {"angle_src": angle_src}
