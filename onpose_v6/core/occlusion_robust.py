"""
가려짐(occlusion)에 강한 키포인트 추출 + 라이브 스무딩.

3단계 안정화:
  1) 2D landmark in-paint  : MediaPipe visibility 낮은 관절은 인접 시점에서 보간
  2) 3D output smoothing   : lifter 출력에 짧은 Savgol 스무딩 (5~7프레임)
  3) Angle EMA + clip      : 최종 각도에 EMA + outlier 클립 (기존 AngleSmoother)

핵심 아이디어:
  - visible 관절 (vis > 0.7)  → MediaPipe 그대로
  - partial   (0.35~0.7)      → 가중 평균
  - occluded  (vis < 0.35)    → 직전 프레임 동일 관절 hold + lifter 우선
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class Landmark2DSmoother:
    """MediaPipe 33 landmark 시간적 스무딩 + occlusion 강건 추출.

    3단계 추정 (우선순위):
      1) visibility 충분 (>= vis_th)        → EMA로 부드럽게
      2) 좌/우 한 쌍 중 한쪽이 visible      → 짝 관절을 좌우 미러로 추정
      3) 직전 visible 값 hold (최대 N프레임)
    """
    # MediaPipe Pose 좌우 쌍 (visible 한 쪽으로 occluded 쪽 추정)
    LR_PAIRS = {
        11: 12, 12: 11,    # shoulder
        13: 14, 14: 13,    # elbow
        15: 16, 16: 15,    # wrist
        17: 18, 18: 17,    # pinky
        19: 20, 20: 19,    # index
        21: 22, 22: 21,    # thumb
        23: 24, 24: 23,    # hip
        25: 26, 26: 25,    # knee
        27: 28, 28: 27,    # ankle
        29: 30, 30: 29,    # heel
        31: 32, 32: 31,    # foot index
    }

    def __init__(self, n_landmarks: int = 33, alpha: float = 0.55,
                 vis_threshold: float = 0.20, hold_max_frames: int = 8,
                 use_mirror: bool = False, max_step: float = 0.08):
        self.n = n_landmarks
        self.alpha = alpha
        self.vis_th = vis_threshold       # 0.20: 정말 안 보일 때만 in-paint
        self.hold_max = hold_max_frames
        self.use_mirror = use_mirror
        self.max_step = max_step          # 한 프레임 최대 이동 (큰 점프 거부)
        self.last_xy: Dict[int, Tuple[float, float, float]] = {}
        self.hold_count: Dict[int, int] = {}
        self._hip_center_x: Optional[float] = None
        self._is_side_view: bool = False   # 옆모습 자동 감지
        self.was_inpainted: Dict[int, bool] = {}

    def reset(self):
        self.last_xy.clear(); self.hold_count.clear()
        self._hip_center_x = None; self._is_side_view = False
        self.was_inpainted.clear()

    def _update_hip_center_and_view(self, landmarks):
        """hip x-center 추적 + 옆모습/정면 자동 판별 (sticky EMA).

        옆모습: 양쪽 어깨/엉덩이 x 차이가 매우 작음 (3D depth 방향으로 겹침).
        한 쪽이 가려진 상태에서도 직전 판정을 유지하도록 sticky 처리.
        """
        v23 = float(getattr(landmarks[23], "visibility", 0.0))
        v24 = float(getattr(landmarks[24], "visibility", 0.0))
        if v23 + v24 > 0.4:
            x = (float(landmarks[23].x) * v23 + float(landmarks[24].x) * v24) / max(v23 + v24, 1e-6)
            self._hip_center_x = x if self._hip_center_x is None \
                                  else 0.7 * self._hip_center_x + 0.3 * x

        v11 = float(getattr(landmarks[11], "visibility", 0.0))
        v12 = float(getattr(landmarks[12], "visibility", 0.0))
        # 옆모습 evidence — 양쪽 모두 visible 일 때만 새 신호 수집
        new_side: Optional[bool] = None
        if v11 > 0.4 and v12 > 0.4 and v23 > 0.3 and v24 > 0.3:
            dx_sh = abs(float(landmarks[11].x) - float(landmarks[12].x))
            dx_hip = abs(float(landmarks[23].x) - float(landmarks[24].x))
            # 어깨와 엉덩이 모두 매우 좁으면 옆모습, 둘 다 충분히 벌어지면 정면
            if dx_sh < 0.06 and dx_hip < 0.05:
                new_side = True
            elif dx_sh > 0.10 or dx_hip > 0.08:
                new_side = False
        # sticky 업데이트: 직전 판정 유지하면서 새 신호로 천천히 전환
        if new_side is not None:
            self._is_side_view = new_side
        # 한 쪽이 가려진 상태면 직전 판정 그대로 유지

    def __call__(self, landmarks):
        if landmarks is None:
            return None
        from types import SimpleNamespace
        self._update_hip_center_and_view(landmarks)
        out = []
        for i, lm in enumerate(landmarks):
            v = float(getattr(lm, "visibility", 1.0))
            x = float(lm.x); y = float(lm.y); z = float(getattr(lm, "z", 0.0))
            inpainted = False
            if v >= self.vis_th:
                # 1) 잘 보임 → EMA (단, 큰 점프는 제한)
                if i in self.last_xy:
                    px, py, pz = self.last_xy[i]
                    dx, dy = x - px, y - py
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag > self.max_step:
                        # outlier — 직전 값에서 max_step 만큼만 이동
                        s = self.max_step / max(mag, 1e-6)
                        x = px + dx * s
                        y = py + dy * s
                    x = self.alpha * x + (1 - self.alpha) * px
                    y = self.alpha * y + (1 - self.alpha) * py
                    z = self.alpha * z + (1 - self.alpha) * pz
                self.last_xy[i] = (x, y, z)
                self.hold_count[i] = 0
            else:
                # 정말 안 보임 (vis < 0.20)
                mirrored = False
                # 2) 옆모습이고 짝 관절이 매우 잘 보일 때만 미러 (정면일 땐 미러 안 함)
                if (self.use_mirror and self._is_side_view and i in self.LR_PAIRS
                        and self._hip_center_x is not None):
                    pair_idx = self.LR_PAIRS[i]
                    pair_lm = landmarks[pair_idx]
                    pv = float(getattr(pair_lm, "visibility", 0.0))
                    if pv >= 0.6:    # 짝 관절은 매우 잘 보여야 미러 활성
                        mx = 2 * self._hip_center_x - float(pair_lm.x)
                        my = float(pair_lm.y)
                        mz = float(getattr(pair_lm, "z", 0.0))
                        # 직전 자기 값과 너무 다르면 거부 (보수적)
                        if i in self.last_xy:
                            px, py, _ = self.last_xy[i]
                            jump = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
                            if jump < self.max_step * 2.5:
                                x = 0.5 * mx + 0.5 * px
                                y = 0.5 * my + 0.5 * py
                                z = mz
                                self.last_xy[i] = (x, y, z)
                                self.hold_count[i] = self.hold_count.get(i, 0) + 1
                                v = max(v, 0.50)
                                mirrored = True
                                inpainted = True
                        else:
                            # 첫 추정 — 미러 그대로 사용
                            x, y, z = mx, my, mz
                            self.last_xy[i] = (x, y, z)
                            v = max(v, 0.45)
                            mirrored = True
                            inpainted = True
                # 3) 미러 불가 → 직전 hold
                if not mirrored:
                    if i in self.last_xy and self.hold_count.get(i, 0) < self.hold_max:
                        x, y, z = self.last_xy[i]
                        self.hold_count[i] = self.hold_count.get(i, 0) + 1
                        v = max(v, 0.35)
                        inpainted = True
                    # 그 외엔 그대로 (실제 vis 그대로 — 화면에선 안 그려짐)
            self.was_inpainted[i] = inpainted
            out.append(SimpleNamespace(x=x, y=y, z=z, visibility=v))
        return out


class Frame3DSmoother:
    """3D lifter 출력 시퀀스에 짧은 윈도우 Savgol 스무딩.

    매 프레임 직전 N프레임을 가지고 마지막 프레임 출력만 스무딩 (causal).
    Savgol은 짧은 윈도우(5~7)면 약간의 지연 + 큰 jitter 완화.
    """
    def __init__(self, window: int = 7, polyorder: int = 2):
        self.window = window if window % 2 == 1 else window + 1
        self.polyorder = min(polyorder, self.window - 1)
        self.buffer: deque = deque(maxlen=self.window)

    def reset(self):
        self.buffer.clear()

    def __call__(self, frame3d: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame3d is None:
            return None
        self.buffer.append(frame3d.astype(np.float32))
        if len(self.buffer) < 3:
            return frame3d
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            # 폴백: 단순 가중 평균
            arr = np.asarray(self.buffer, dtype=np.float32)
            weights = np.linspace(0.5, 1.0, len(self.buffer))
            weights /= weights.sum()
            return np.einsum("t,tjc->jc", weights, arr)
        arr = np.asarray(self.buffer, dtype=np.float32)
        n = len(arr)
        w = min(self.window, n if n % 2 == 1 else n - 1)
        if w < self.polyorder + 2:
            return arr[-1]
        smoothed = np.empty_like(arr)
        for j in range(arr.shape[1]):
            for c in range(arr.shape[2]):
                smoothed[:, j, c] = savgol_filter(arr[:, j, c], w, self.polyorder, mode="interp")
        return smoothed[-1]


class OcclusionAwareJointBlender:
    """3D lifter 출력을 visibility 기반으로 보정.

    visible joint:   lifter 출력 그대로
    partially visible: lifter * 0.6 + previous * 0.4
    occluded:        previous frame 값 hold (in-paint)
    """
    def __init__(self):
        self.last: Optional[np.ndarray] = None

    def reset(self):
        self.last = None

    def __call__(self, frame3d: np.ndarray, visibilities: List[float]) -> np.ndarray:
        if frame3d is None:
            return frame3d
        cur = frame3d.copy()
        if self.last is None:
            self.last = cur.copy()
            return cur
        for j, v in enumerate(visibilities):
            if j >= len(cur):
                continue
            if v < 0.25:
                # 매우 가려짐 → 직전 값 + bone-length 유지 (lifter는 신뢰 어려움)
                cur[j] = self.last[j]
            elif v < 0.55:
                # 부분 가려짐 → 가중 평균
                cur[j] = 0.55 * cur[j] + 0.45 * self.last[j]
        self.last = cur.copy()
        return cur


class FrameRecorder:
    """캡처 중 사용자 영상 프레임을 buffer에 저장 — 결과 화면 리플레이용.

    메모리 절약을 위해:
      - 최대 N프레임 (5초 * 30fps = 150)
      - resize해서 저장 (480x270 정도)
    """
    def __init__(self, max_frames: int = 200, target_size: Tuple[int, int] = (640, 360)):
        self.max_frames = max_frames
        self.size = target_size
        self.frames: list = []

    def reset(self):
        self.frames = []

    def add(self, bgr_frame: np.ndarray) -> None:
        import cv2
        if bgr_frame is None:
            return
        if len(self.frames) >= self.max_frames:
            return
        h, w = bgr_frame.shape[:2]
        if (w, h) != self.size:
            bgr_frame = cv2.resize(bgr_frame, self.size)
        self.frames.append(bgr_frame)

    def get_loop_frame(self, t_index: int) -> Optional[np.ndarray]:
        if not self.frames:
            return None
        return self.frames[t_index % len(self.frames)]

    def count(self) -> int:
        return len(self.frames)
