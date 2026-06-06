"""
Test-Time Augmentation (TTA) — 추론 시점에 입력을 약간 변형해 여러 번 추론한 뒤 평균.

가려짐 대처에 가장 직접적인 효과:
- 좌우 반전 (L-R flip): 오른쪽이 가려진 자세도 좌우 반전 후 재추론하면
  '왼쪽이 가려진 자세'로 보여서 lifter가 다른 시각에서 추론. 두 결과를 평균.

원리:
  pred(x) ≈ E[pred(aug(x))] 이라면, augmentation 평균이 noise를 줄임.
  특히 occlusion이 비대칭일 때 효과 큼.

사용:
  from core.tta import lr_flip_tta_sequence
  smoothed = lr_flip_tta_sequence(pred_seq, alpha=0.5)
"""
from __future__ import annotations

from typing import List

import numpy as np

# JOINT_ORDER (dataset.py 기준)
JOINT_NAMES = [
    "Head", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle", "Hip",
]
JOINT_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# 좌우 대응 쌍
LR_SWAP_PAIRS = [
    ("LShoulder", "RShoulder"),
    ("LElbow", "RElbow"),
    ("LWrist", "RWrist"),
    ("LHip", "RHip"),
    ("LKnee", "Rknee"),
    ("LAnkle", "RAnkle"),
]


def flip_skeleton_lr(seq: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    골격의 좌우를 반전.
    - sagittal_axis (X)에 대해 부호 반전
    - L/R 관절 쌍을 인덱스 스왑

    Args:
        seq: (T, J, 3)
        axis: 좌우 축 (root-centered 좌표 기준, X축=0이 일반적)
    Returns:
        (T, J, 3) 좌우 반전된 골격
    """
    out = seq.copy()
    out[..., axis] *= -1.0
    for l_name, r_name in LR_SWAP_PAIRS:
        l, r = JOINT_IDX[l_name], JOINT_IDX[r_name]
        tmp = out[:, l].copy()
        out[:, l] = out[:, r]
        out[:, r] = tmp
    return out


def lr_flip_tta_sequence(pred_seq: np.ndarray, pred_seq_flipped: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """
    원본 입력의 lifter 결과 + 좌우 반전한 입력의 lifter 결과를 평균.

    Args:
        pred_seq: (T, J, 3) 원본 입력으로 lifter가 출력한 3D
        pred_seq_flipped: (T, J, 3) 좌우 반전 입력으로 lifter가 출력한 3D
                          (이미 다시 unflip된 좌표)
        alpha: 원본 가중치 (default 0.5 = 동등 평균)
    Returns:
        (T, J, 3) TTA 평균
    """
    # 두 lifter 출력 차원이 다르면 잘라 맞춤
    n = min(len(pred_seq), len(pred_seq_flipped))
    avg = alpha * pred_seq[:n] + (1 - alpha) * pred_seq_flipped[:n]
    return avg.astype(np.float32)


def landmarks_lr_flipped(landmarks, image_w: int = 1) -> list:
    """
    MediaPipe 33-keypoint landmarks를 좌우 반전.
    Visibility는 그대로 유지하되 좌우 인덱스를 스왑하고 x값을 반전.

    MediaPipe Pose 좌우 쌍:
      (1,4) 눈 / (2,5) 눈 / (3,6) 귀 / (7,8) 귀
      (11,12) 어깨 / (13,14) 팔꿈치 / (15,16) 손목
      (17,18) 새끼 / (19,20) 검지 / (21,22) 엄지
      (23,24) 골반 / (25,26) 무릎 / (27,28) 발목
      (29,30) 발뒤꿈치 / (31,32) 발끝
    """
    MP_LR_SWAP = {
        1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3, 7: 8, 8: 7,
        11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,
        17: 18, 18: 17, 19: 20, 20: 19, 21: 22, 22: 21,
        23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
        29: 30, 30: 29, 31: 32, 32: 31,
    }
    # 단순 SimpleNamespace 객체로 변환 (mediapipe Landmark과 같은 인터페이스)
    from types import SimpleNamespace
    flipped = []
    for i in range(len(landmarks)):
        src_i = MP_LR_SWAP.get(i, i)
        src = landmarks[src_i]
        flipped.append(SimpleNamespace(
            x=1.0 - float(src.x),       # 좌우 반전
            y=float(src.y),
            z=getattr(src, "z", 0.0),
            visibility=getattr(src, "visibility", 1.0),
        ))
    return flipped
