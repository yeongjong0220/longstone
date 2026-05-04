from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from dataset import (
    JOINT_ORDER,
    PHASE_NAMES_3,
    PHASE_NAMES_4,
    coarse_progress_phase_labels,
    cyclic_anchor_phase_labels,
    normalize_skeleton,
    read_keypoint_csv,
)


JOINT_IDX = {name: idx for idx, name in enumerate(JOINT_ORDER)}


def moving_average_1d(values: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or len(values) == 0:
        return values.astype(np.float64, copy=False)
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    filt = np.ones(kernel, dtype=np.float64) / float(kernel)
    return np.convolve(padded, filt, mode="valid")


def angle_deg(seq3d: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    p1 = seq3d[:, JOINT_IDX[a], :]
    p2 = seq3d[:, JOINT_IDX[b], :]
    p3 = seq3d[:, JOINT_IDX[c], :]
    u = p1 - p2
    v = p3 - p2
    u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-8, None)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)
    cos = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def first_difference(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.concatenate([[0.0], np.diff(values).astype(np.float64)])


def extract_kinematic_features(
    seq3d: np.ndarray,
    feature_set: str = "hip_knee",
    smooth_kernel: int = 5,
) -> pd.DataFrame:
    seq3d = np.nan_to_num(np.asarray(seq3d, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    left_hip = moving_average_1d(angle_deg(seq3d, "Neck", "LHip", "LKnee"), smooth_kernel)
    right_hip = moving_average_1d(angle_deg(seq3d, "Neck", "RHip", "Rknee"), smooth_kernel)
    left_knee = moving_average_1d(angle_deg(seq3d, "LHip", "LKnee", "LAnkle"), smooth_kernel)
    right_knee = moving_average_1d(angle_deg(seq3d, "RHip", "Rknee", "RAnkle"), smooth_kernel)
    trunk = angle_deg(seq3d, "Hip", "Neck", "Head")

    hip = moving_average_1d((left_hip + right_hip) / 2.0, smooth_kernel)
    knee = moving_average_1d((left_knee + right_knee) / 2.0, smooth_kernel)
    trunk = moving_average_1d(trunk, smooth_kernel)
    shoulder_wrist_y_delta = moving_average_1d(
        (
            np.abs(seq3d[:, JOINT_IDX["LShoulder"], 1] - seq3d[:, JOINT_IDX["LWrist"], 1])
            + np.abs(seq3d[:, JOINT_IDX["RShoulder"], 1] - seq3d[:, JOINT_IDX["RWrist"], 1])
        )
        / 2.0,
        smooth_kernel,
    )
    arm_reach = moving_average_1d(
        (
            np.linalg.norm(seq3d[:, JOINT_IDX["LShoulder"], :] - seq3d[:, JOINT_IDX["LWrist"], :], axis=1)
            + np.linalg.norm(seq3d[:, JOINT_IDX["RShoulder"], :] - seq3d[:, JOINT_IDX["RWrist"], :], axis=1)
        )
        / 2.0,
        smooth_kernel,
    )
    hip_symmetry = moving_average_1d(
        np.abs(seq3d[:, JOINT_IDX["LHip"], 1] - seq3d[:, JOINT_IDX["RHip"], 1]),
        smooth_kernel,
    )
    knee_symmetry = moving_average_1d(
        np.abs(seq3d[:, JOINT_IDX["LKnee"], 1] - seq3d[:, JOINT_IDX["Rknee"], 1]),
        smooth_kernel,
    )

    data = {
        "hip_angle": hip,
        "knee_angle": knee,
        "hip_velocity": moving_average_1d(first_difference(hip), smooth_kernel),
        "knee_velocity": moving_average_1d(first_difference(knee), smooth_kernel),
    }
    if feature_set == "hip_knee_trunk":
        data["trunk_angle"] = trunk
        data["trunk_velocity"] = moving_average_1d(first_difference(trunk), smooth_kernel)
    elif feature_set == "lower_body_full":
        data = {
            "left_hip_angle": left_hip,
            "right_hip_angle": right_hip,
            "left_knee_angle": left_knee,
            "right_knee_angle": right_knee,
            "trunk_angle": trunk,
            "hip_velocity": moving_average_1d(first_difference(hip), smooth_kernel),
            "knee_velocity": moving_average_1d(first_difference(knee), smooth_kernel),
            "trunk_velocity": moving_average_1d(first_difference(trunk), smooth_kernel),
        }
    elif feature_set == "spine_stretch":
        data = {
            "hip_angle": hip,
            "knee_angle": knee,
            "trunk_angle": trunk,
            "arm_reach": arm_reach,
            "shoulder_wrist_y_delta": shoulder_wrist_y_delta,
            "hip_velocity": moving_average_1d(first_difference(hip), smooth_kernel),
            "knee_velocity": moving_average_1d(first_difference(knee), smooth_kernel),
            "trunk_velocity": moving_average_1d(first_difference(trunk), smooth_kernel),
        }
    elif feature_set == "bridging":
        data = {
            "hip_angle": hip,
            "knee_angle": knee,
            "trunk_angle": trunk,
            "hip_symmetry": hip_symmetry,
            "knee_symmetry": knee_symmetry,
            "hip_velocity": moving_average_1d(first_difference(hip), smooth_kernel),
            "knee_velocity": moving_average_1d(first_difference(knee), smooth_kernel),
            "trunk_velocity": moving_average_1d(first_difference(trunk), smooth_kernel),
        }
    elif feature_set != "hip_knee":
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return pd.DataFrame(data)


FEATURE_SET_CHOICES = [
    "hip_knee",
    "hip_knee_trunk",
    "lower_body_full",
    "spine_stretch",
    "bridging",
]


def phase_names_for_scheme(phase_scheme: str) -> List[str]:
    if phase_scheme == "cyclic4":
        return list(PHASE_NAMES_4)
    if phase_scheme == "progress3":
        return list(PHASE_NAMES_3)
    raise ValueError(f"Unknown phase_scheme: {phase_scheme}")


def phase_labels_for_sequence(seq3d: np.ndarray, phase_scheme: str) -> np.ndarray:
    if phase_scheme == "cyclic4":
        return cyclic_anchor_phase_labels(seq3d)
    if phase_scheme == "progress3":
        return coarse_progress_phase_labels(seq3d)
    raise ValueError(f"Unknown phase_scheme: {phase_scheme}")


def load_normalized_3d_csv(csv_path, start_frame: int | None = None, end_frame: int | None = None) -> np.ndarray:
    seq = read_keypoint_csv(csv_path, dims=3)
    if start_frame is not None or end_frame is not None:
        start = max(0, int(start_frame or 0))
        end = min(len(seq) - 1, int(end_frame if end_frame is not None else len(seq) - 1))
        seq = seq[start : end + 1]
    return normalize_skeleton(seq)


def regularized_covariance(
    x: np.ndarray,
    covariance_mode: str = "full",
    shrinkage: float = 0.15,
    eps: float = 1e-3,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dim = x.shape[1]
    if x.shape[0] <= 1:
        cov = np.eye(dim, dtype=np.float64)
    else:
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov).astype(np.float64)

    diag = np.diag(np.clip(np.diag(cov), eps, None))
    if covariance_mode == "diagonal":
        cov = diag
    elif covariance_mode == "full":
        shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
    else:
        raise ValueError(f"Unknown covariance_mode: {covariance_mode}")

    cov += np.eye(dim, dtype=np.float64) * float(eps)
    return cov


def mahalanobis_d2(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)
    return np.einsum("bi,ij,bj->b", delta, inv_cov, delta)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) == 0:
        return values
    out = np.empty_like(values)
    half = int(window) // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def status_from_score(score: float, pass_threshold: float, warn_threshold: float) -> str:
    if score <= pass_threshold:
        return "PASS"
    if score <= warn_threshold:
        return "WARN"
    return "FAIL"
