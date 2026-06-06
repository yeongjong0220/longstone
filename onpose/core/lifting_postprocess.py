"""
2D->3D Lifting 정확도를 끌어올리는 후처리 모듈.

진단 결과 기반 (causal lifter on The Seal):
- Knee 위치 오차가 hip의 2~4배 (꺾이는 관절 + 가려짐)
- Right side > Left side (특정 자세에서 오른쪽 가려짐)
- MIDDLE phase 오차가 END phase의 10배 (동작 전환부 jitter)
- 출력 3D의 다리뼈 길이가 시간에 따라 들쭉날쭉 (std/mean 0.29 vs GT 0.16)

대응:
  1) temporal_smooth_3d   - 시간 축 Savitzky-Golay/1D Gaussian (jitter 완화)
  2) enforce_bone_length  - 토르소 길이를 anchor로 다른 bone을 평균 비율로 강제
  3) lr_symmetry_blend    - 좌우 대칭 자세에서 가려진 쪽을 보이는 쪽으로 부분 보강
  4) ensemble_average     - 여러 lifter 출력의 가중 평균 (TTA용)
  5) confidence_weighted  - visibility/error 추정값으로 가중

모두 추론 후 적용 — 재학습 불필요.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# -----------------------------------------------------------------------------
# 1) Temporal smoothing
# -----------------------------------------------------------------------------
def gaussian_1d_kernel(window: int, sigma: float) -> np.ndarray:
    half = (window - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def temporal_smooth_3d(seq: np.ndarray, method: str = "savgol",
                       window: int = 9, polyorder: int = 3,
                       sigma: float = 2.0) -> np.ndarray:
    """
    Args:
        seq: (T, J, 3)
        method: "savgol" (Savitzky-Golay) | "gaussian" | "median"
    Returns:
        smoothed (T, J, 3)
    """
    if len(seq) < window:
        return seq
    if method == "savgol" and _HAS_SCIPY:
        # Savitzky-Golay: outlier에 강하고 골격 형태 유지
        out = np.empty_like(seq)
        win = window if window % 2 == 1 else window + 1
        win = min(win, len(seq) - (1 - len(seq) % 2))   # 시퀀스 길이 미만 + 홀수
        if win < polyorder + 2:
            return seq
        for j in range(seq.shape[1]):
            for c in range(seq.shape[2]):
                out[:, j, c] = savgol_filter(seq[:, j, c], win, polyorder, mode="interp")
        return out
    if method == "gaussian":
        k = gaussian_1d_kernel(window, sigma)
        out = seq.copy()
        pad = (window - 1) // 2
        for j in range(seq.shape[1]):
            for c in range(seq.shape[2]):
                series = np.pad(seq[:, j, c], pad, mode="edge")
                out[:, j, c] = np.convolve(series, k, mode="valid")
        return out
    if method == "median":
        from scipy.ndimage import median_filter
        out = seq.copy()
        for j in range(seq.shape[1]):
            for c in range(seq.shape[2]):
                out[:, j, c] = median_filter(seq[:, j, c], size=window)
        return out
    return seq


# -----------------------------------------------------------------------------
# 2) Bone-length consistency
# -----------------------------------------------------------------------------
# JOINT_ORDER (dataset.py 기준) 의 인덱스
JOINT_NAMES = [
    "Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle", "Hip",
]
JOINT_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# 인체 골격 정의 (이 모델의 관절 토폴로지)
BONES = [
    ("Head", "Neck"),
    ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("Neck", "Hip"),
    ("Hip", "LHip"), ("Hip", "RHip"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "Rknee"), ("Rknee", "RAnkle"),
]


def enforce_bone_length(seq: np.ndarray, anchor: Tuple[str, str] = ("Neck", "Hip"),
                       use_target_bones: Optional[Dict[Tuple[str, str], float]] = None
                       ) -> np.ndarray:
    """
    각 프레임의 모든 bone 길이를 anchor (예: 토르소 Neck-Hip) 길이의 일정 비율로 강제.

    아이디어:
      - 시퀀스 전체의 평균 bone length 비율을 계산 (anchor 대비)
      - 매 프레임에서 자식 관절을 부모 관절 + 단위벡터*평균길이 로 재구성

    Args:
        seq: (T, J, 3)
        anchor: 길이가 가장 안정적인 bone (default Neck-Hip = 토르소)
        use_target_bones: 외부에서 알려진 정답 bone 비율 (옵션)
    Returns:
        corrected (T, J, 3)
    """
    if seq.size == 0:
        return seq
    out = seq.copy().astype(np.float32)
    a_idx, b_idx = JOINT_IDX[anchor[0]], JOINT_IDX[anchor[1]]
    anchor_lens = np.linalg.norm(seq[:, a_idx] - seq[:, b_idx], axis=-1)
    anchor_ref = float(np.median(anchor_lens))   # 중앙값 (outlier 견고)

    # 각 bone의 평균 길이/anchor 비율
    bone_ratios = {}
    for parent, child in BONES:
        p, c = JOINT_IDX[parent], JOINT_IDX[child]
        lens = np.linalg.norm(seq[:, p] - seq[:, c], axis=-1)
        # median으로 outlier 영향 줄임
        bone_ratios[(parent, child)] = float(np.median(lens)) / max(anchor_ref, 1e-8)
    if use_target_bones:
        bone_ratios.update(use_target_bones)

    # Hip 기준으로 forward kinematics 식의 단순 보정
    # Hip → (LHip, RHip, Neck) → (… → 끝 관절) 트리 순서로 재구성
    parent_chain = [
        ("Hip", "LHip"), ("Hip", "RHip"), ("Hip", "Neck"),
        ("Neck", "LShoulder"), ("Neck", "RShoulder"), ("Neck", "Head"),
        ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
        ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
        ("LHip", "LKnee"), ("LKnee", "LAnkle"),
        ("RHip", "Rknee"), ("Rknee", "RAnkle"),
    ]
    # 각 frame에 대해 다시 그림
    for t in range(len(out)):
        anchor_len_t = anchor_ref     # 토르소를 절대 기준 길이로 고정
        # 1) 먼저 Hip-Neck을 정확히 anchor_ref로 맞춤
        if anchor == ("Neck", "Hip"):
            cur_vec = out[t, JOINT_IDX["Neck"]] - out[t, JOINT_IDX["Hip"]]
            cur_n = np.linalg.norm(cur_vec)
            if cur_n > 1e-8:
                out[t, JOINT_IDX["Neck"]] = out[t, JOINT_IDX["Hip"]] + cur_vec / cur_n * anchor_len_t
        # 2) 나머지 bone을 비율에 맞춰 부모 + 단위벡터*target_len 로
        for parent, child in parent_chain:
            p, c = JOINT_IDX[parent], JOINT_IDX[child]
            vec = out[t, c] - out[t, p]
            n = np.linalg.norm(vec)
            if n < 1e-8:
                continue
            target_len = bone_ratios.get((parent, child),
                                         bone_ratios.get((child, parent), 1.0)) * anchor_len_t
            out[t, c] = out[t, p] + vec / n * target_len
    return out


# -----------------------------------------------------------------------------
# 3) Left-Right symmetry blend
# -----------------------------------------------------------------------------
LR_PAIRS = [("LShoulder", "RShoulder"), ("LElbow", "RElbow"), ("LWrist", "RWrist"),
            ("LHip", "RHip"), ("LKnee", "Rknee"), ("LAnkle", "RAnkle")]


def lr_symmetry_blend(seq: np.ndarray, alpha: float = 0.5,
                     visibility_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    좌우 대칭 자세(예: The Seal, Bridging)에서 가려진 쪽을 보이는 쪽 미러로 보강.

    Args:
        seq: (T, J, 3)
        alpha: 0.0 = 미러만, 1.0 = 원본만 사용 (default 0.5)
        visibility_mask: (T, J) — 1 = 잘 보임, 0 = 가려짐 (해당 시점은 alpha 작게)
    """
    out = seq.copy()
    sagittal_axis = 0   # X축이 좌우라 가정 (root-centered)
    for l_name, r_name in LR_PAIRS:
        l, r = JOINT_IDX[l_name], JOINT_IDX[r_name]
        # 좌우 미러: y, z는 같고 x는 부호 반전
        mirrored_l = seq[:, r].copy()
        mirrored_l[..., sagittal_axis] *= -1.0
        mirrored_r = seq[:, l].copy()
        mirrored_r[..., sagittal_axis] *= -1.0
        if visibility_mask is not None:
            # 가려진 쪽일수록 미러 비중 ↑
            v_l = visibility_mask[:, l, None]
            v_r = visibility_mask[:, r, None]
            a_l = np.clip(alpha + (1 - v_l) * (1 - alpha), 0, 1)
            a_r = np.clip(alpha + (1 - v_r) * (1 - alpha), 0, 1)
            out[:, l] = a_l * seq[:, l] + (1 - a_l) * mirrored_l
            out[:, r] = a_r * seq[:, r] + (1 - a_r) * mirrored_r
        else:
            out[:, l] = alpha * seq[:, l] + (1 - alpha) * mirrored_l
            out[:, r] = alpha * seq[:, r] + (1 - alpha) * mirrored_r
    return out


# -----------------------------------------------------------------------------
# 4) Ensemble (multi-lifter average)
# -----------------------------------------------------------------------------
def ensemble_average(seqs: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None
                    ) -> np.ndarray:
    """여러 lifter 결과를 가중 평균"""
    if weights is None:
        weights = [1.0] * len(seqs)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    out = np.zeros_like(seqs[0])
    for s, w in zip(seqs, weights):
        out = out + w * s
    return out


# -----------------------------------------------------------------------------
# 5) Outlier robustification (per-frame velocity cap)
# -----------------------------------------------------------------------------
def clip_velocity(seq: np.ndarray, max_step: float = 0.10) -> np.ndarray:
    """
    프레임 사이 변화가 너무 큰 spike를 직전 프레임 방향으로 제한.
    max_step: root-normalized 단위에서 한 프레임 최대 이동 거리.
    """
    if len(seq) < 2:
        return seq
    out = seq.copy().astype(np.float32)
    for t in range(1, len(out)):
        delta = out[t] - out[t - 1]
        mag = np.linalg.norm(delta, axis=-1, keepdims=True)
        scale = np.where(mag > max_step, max_step / np.clip(mag, 1e-8, None), 1.0)
        out[t] = out[t - 1] + delta * scale
    return out


# -----------------------------------------------------------------------------
# 6) Composite pipeline — 권장 조합
# -----------------------------------------------------------------------------
def postprocess_pipeline(seq: np.ndarray,
                         smooth_window: int = 9,
                         do_bone_lock: bool = True,
                         do_velocity_clip: bool = True,
                         max_velocity_step: float = 0.10) -> np.ndarray:
    """
    추천 후처리 순서:
        outlier velocity clip → bone-length lock → temporal smoothing
    """
    out = seq.astype(np.float32, copy=True)
    if do_velocity_clip and len(out) > 1:
        out = clip_velocity(out, max_step=max_velocity_step)
    if do_bone_lock:
        out = enforce_bone_length(out)
    if smooth_window > 1:
        out = temporal_smooth_3d(out, method="savgol", window=smooth_window, polyorder=3)
    return out


# -----------------------------------------------------------------------------
# DataFrame helpers (CSV 입출력 호환)
# -----------------------------------------------------------------------------
def df_to_seq(df) -> np.ndarray:
    """CSV DataFrame → (T, J, 3) ndarray"""
    arrs = []
    for j in JOINT_NAMES:
        arrs.append(df[[f"{j}_x", f"{j}_y", f"{j}_z"]].to_numpy(dtype=float))
    return np.stack(arrs, axis=1)


def seq_to_df(seq: np.ndarray, base_df=None):
    import pandas as pd
    rows = {}
    if base_df is not None and "frame" in base_df.columns:
        rows["frame"] = base_df["frame"].to_numpy()[:len(seq)]
    if base_df is not None and "phase_idx" in base_df.columns:
        rows["phase_idx"] = base_df["phase_idx"].to_numpy()[:len(seq)]
    for i, j in enumerate(JOINT_NAMES):
        for k, ax in enumerate(["x", "y", "z"]):
            rows[f"{j}_{ax}"] = seq[:, i, k]
    return pd.DataFrame(rows)
