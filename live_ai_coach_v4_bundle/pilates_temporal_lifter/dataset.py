from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch 없는 환경에서 manifest 생성용
    torch = None

    class Dataset:  # type: ignore[misc]
        pass


JOINT_ORDER: List[str] = [
    "Head",
    "Neck",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "Rknee",
    "LAnkle",
    "RAnkle",
    "Hip",
]

BONES: List[Tuple[str, str]] = [
    ("Head", "Neck"),
    ("Neck", "LShoulder"),
    ("Neck", "RShoulder"),
    ("LShoulder", "LElbow"),
    ("RShoulder", "RElbow"),
    ("LElbow", "LWrist"),
    ("RElbow", "RWrist"),
    ("Neck", "Hip"),
    ("Hip", "LHip"),
    ("Hip", "RHip"),
    ("LHip", "LKnee"),
    ("RHip", "Rknee"),
    ("LKnee", "LAnkle"),
    ("Rknee", "RAnkle"),
]

PHASE_NAMES_3 = ["START", "MIDDLE", "END"]
PHASE_NAMES_4 = ["READY", "OUTBOUND", "APEX", "RETURN"]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "이 단계는 PyTorch가 필요합니다. 먼저 manifest 생성은 torch 없이 가능하지만, 학습/추론 전에는 torch를 설치하세요."
        )


def _safe_json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _moving_average(x: np.ndarray, kernel: int = 5) -> np.ndarray:
    if kernel <= 1 or x.size == 0:
        return x
    pad = kernel // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    filt = np.ones(kernel, dtype=np.float32) / float(kernel)
    return np.convolve(padded, filt, mode="valid")


def _extract_signature(name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    actor = None
    timestamp = None
    camera = None

    actor_match = re.search(r"(actorP\d+)", name)
    if actor_match:
        actor = actor_match.group(1)

    ts_match = re.search(r"(\d{8}_\d{2}\.\d{2}\.\d{2})", name)
    if ts_match:
        timestamp = ts_match.group(1)

    cam_match = re.search(r"_CAM_(\d+)", name)
    if cam_match:
        camera = int(cam_match.group(1))

    return actor, timestamp, camera


def canonical_clip_id(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_CAM_\d+$", "", stem)
    return stem


def read_keypoint_csv(path: Path, dims: int) -> np.ndarray:
    df = pd.read_csv(path)
    frames = []
    for joint in JOINT_ORDER:
        cols = [f"{joint}_x", f"{joint}_y"]
        if dims == 3:
            cols.append(f"{joint}_z")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{path} 에 필요한 컬럼이 없습니다: {missing}")
        frames.append(df[cols].to_numpy(dtype=np.float32))
    arr = np.stack(frames, axis=1).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def normalize_skeleton(
    coords: np.ndarray,
    root_joint: str = "Hip",
    shoulder_left: str = "LShoulder",
    shoulder_right: str = "RShoulder",
    neck_joint: str = "Neck",
) -> np.ndarray:
    arr = np.nan_to_num(coords.astype(np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    idx = {name: i for i, name in enumerate(JOINT_ORDER)}
    root = arr[:, idx[root_joint]: idx[root_joint] + 1, :]
    arr = arr - root

    shoulder_width = np.linalg.norm(
        arr[:, idx[shoulder_left], :] - arr[:, idx[shoulder_right], :], axis=-1
    )
    torso = np.linalg.norm(arr[:, idx[neck_joint], :] - arr[:, idx[root_joint], :], axis=-1)
    hip_width = np.linalg.norm(arr[:, idx["LHip"], :] - arr[:, idx["RHip"], :], axis=-1)

    shoulder_width = np.nan_to_num(shoulder_width, nan=0.0, posinf=0.0, neginf=0.0)
    torso = np.nan_to_num(torso, nan=0.0, posinf=0.0, neginf=0.0)
    hip_width = np.nan_to_num(hip_width, nan=0.0, posinf=0.0, neginf=0.0)

    scale = np.nanmedian(np.maximum.reduce([shoulder_width, torso, hip_width]))
    if not np.isfinite(scale) or float(scale) < 1e-6:
        scale = 1.0
    arr /= float(scale)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_observation_mask(x2d: np.ndarray) -> np.ndarray:
    x2d = np.asarray(x2d, dtype=np.float32)
    mask = np.ones((*x2d.shape[:2], 1), dtype=np.float32)
    finite = np.isfinite(x2d).all(axis=-1, keepdims=True)
    near_zero = (np.abs(np.nan_to_num(x2d, nan=0.0, posinf=0.0, neginf=0.0)).sum(axis=-1, keepdims=True) < 1e-8)
    mask[~finite] = 0.0
    mask[near_zero] = 0.0
    return mask


def apply_occlusion_augmentation(
    x2d: np.ndarray,
    rng: np.random.Generator,
    joint_drop_prob: float = 0.03,
    block_prob: float = 0.30,
    max_block_fraction: float = 0.20,
    jitter_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.nan_to_num(x2d.astype(np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = build_observation_mask(out)
    out[mask[..., 0] == 0.0] = 0.0

    joint_drop = rng.random(out.shape[:2]) < joint_drop_prob
    out[joint_drop] = 0.0
    mask[joint_drop, 0] = 0.0

    if rng.random() < block_prob:
        num_joints = int(rng.integers(1, max(2, out.shape[1] // 3)))
        joints = rng.choice(out.shape[1], size=num_joints, replace=False)
        for j in joints:
            span = int(max(2, rng.integers(2, max(3, int(out.shape[0] * max_block_fraction) + 1))))
            start = int(rng.integers(0, max(1, out.shape[0] - span + 1)))
            out[start:start + span, j, :] = 0.0
            mask[start:start + span, j, 0] = 0.0

    out += rng.normal(0.0, jitter_std, size=out.shape).astype(np.float32) * mask
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, mask


def coarse_progress_phase_labels(seq3d: np.ndarray) -> np.ndarray:
    t = seq3d.shape[0]
    if t <= 1:
        return np.zeros((t,), dtype=np.int64)

    vel = np.linalg.norm(np.diff(seq3d, axis=0), axis=-1).mean(axis=-1)
    vel = np.concatenate([vel[:1], vel], axis=0)
    vel = _moving_average(vel, kernel=5)

    cumulative = np.cumsum(np.maximum(vel, 1e-6))
    cumulative = cumulative / max(float(cumulative[-1]), 1e-6)

    labels = np.zeros((t,), dtype=np.int64)
    labels[cumulative >= (1.0 / 3.0)] = 1
    labels[cumulative >= (2.0 / 3.0)] = 2
    return labels


def cyclic_anchor_phase_labels(seq3d: np.ndarray) -> np.ndarray:
    t = seq3d.shape[0]
    if t <= 8:
        return np.zeros((t,), dtype=np.int64)

    start_len = max(5, t // 12)
    start_anchor = seq3d[:start_len].mean(axis=0, keepdims=True)

    dist = np.linalg.norm(seq3d - start_anchor, axis=-1).mean(axis=-1)
    dist = _moving_average(dist, kernel=7)

    ready_thr = float(np.percentile(dist, 20))
    peak = int(np.argmax(dist))
    peak_val = float(dist[peak])

    # 기본은 OUTBOUND
    labels = np.full((t,), 1, dtype=np.int64)

    # 1) 앞쪽 READY: "연속된 prefix"만 READY로 둔다
    prefix_end = start_len - 1
    while prefix_end + 1 < t and dist[prefix_end + 1] <= ready_thr:
        prefix_end += 1
    labels[:prefix_end + 1] = 0  # READY

    # 2) 뒤쪽 READY: "연속된 suffix"만 READY로 둔다
    suffix_start = t
    i = t - 1
    while i >= 0 and dist[i] <= ready_thr:
        i -= 1
    suffix_start = i + 1
    if suffix_start < t:
        labels[suffix_start:] = 0  # READY

    # 3) APEX: peak 주변에서 충분히 큰 dist 구간
    apex_thr = max(float(np.percentile(dist, 85)), 0.85 * peak_val)

    left = peak
    while left - 1 > prefix_end and dist[left - 1] >= apex_thr:
        left -= 1

    right_limit = suffix_start - 1 if suffix_start < t else t - 1
    right = peak
    while right + 1 <= right_limit and dist[right + 1] >= apex_thr:
        right += 1

    labels[left:right + 1] = 2  # APEX

    # 4) RETURN: apex 이후 ~ final READY 이전
    return_start = right + 1
    return_end = suffix_start if suffix_start < t else t
    if return_start < return_end:
        labels[return_start:return_end] = 3  # RETURN

    # 5) OUTBOUND는 READY와 APEX 사이에 남은 구간
    # 이미 기본값이 1이므로 따로 안 건드려도 됨

    return labels


@dataclass
class ClipEntry:
    clip_id: str
    actor: str
    timestamp: str
    camera: int
    csv_2d: str
    csv_3d: str
    json_path: Optional[str]
    start_frame: int
    end_frame: int
    category_1: str
    category_2: str
    level: str


def scan_aihub_pilates_pairs(root: Path) -> List[ClipEntry]:
    root = Path(root)
    csv_files = list(root.rglob("keypoints_*.csv"))
    json_files = list(root.rglob("*.json"))

    three_d_index: Dict[Tuple[Optional[str], Optional[str]], Path] = {}
    two_d_index: Dict[Tuple[Optional[str], Optional[str]], Dict[int, Path]] = {}
    json_index: Dict[Tuple[Optional[str], Optional[str]], Path] = {}

    for path in csv_files:
        actor, ts, cam = _extract_signature(path.name)
        key = (actor, ts)
        if cam is None:
            three_d_index[key] = path
        else:
            two_d_index.setdefault(key, {})[cam] = path

    for path in json_files:
        actor, ts, _ = _extract_signature(path.name)
        key = (actor, ts)
        json_index[key] = path

    entries: List[ClipEntry] = []
    for key, csv3d in sorted(three_d_index.items()):
        if key not in two_d_index:
            continue
        meta = {}
        json_path = json_index.get(key)
        if json_path is not None:
            try:
                meta = _safe_json_load(json_path)
            except Exception:
                meta = {}

        ann = meta.get("annotations", {}) if isinstance(meta, dict) else {}
        actor_meta = meta.get("actor", {}) if isinstance(meta, dict) else {}

        category_1 = str(ann.get("category_1", "unknown"))
        category_2 = str(ann.get("category_2", "unknown"))
        level = str(actor_meta.get("level", "unknown"))
        start_frame = int(ann.get("start_frame", 0))
        end_frame = int(ann.get("end_frame", max(0, len(pd.read_csv(csv3d)) - 1)))

        for camera, csv2d in sorted(two_d_index[key].items()):
            entries.append(
                ClipEntry(
                    clip_id=canonical_clip_id(csv3d),
                    actor=key[0] or "unknown_actor",
                    timestamp=key[1] or "unknown_time",
                    camera=camera,
                    csv_2d=str(csv2d),
                    csv_3d=str(csv3d),
                    json_path=str(json_path) if json_path else None,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    category_1=category_1,
                    category_2=category_2,
                    level=level,
                )
            )
    return entries


def save_manifest(entries: Sequence[ClipEntry], path: Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> List[ClipEntry]:
    entries: List[ClipEntry] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(ClipEntry(**json.loads(line)))
    return entries


def split_by_actor(
    entries: Sequence[ClipEntry],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> Tuple[List[ClipEntry], List[ClipEntry], List[ClipEntry]]:
    actors = sorted({e.actor for e in entries})
    n = len(actors)
    n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_ratio))) if n >= 5 else max(0, min(1, n - n_test - 1))
    test_actors = set(actors[:n_test])
    val_actors = set(actors[n_test:n_test + n_val])

    train, val, test = [], [], []
    for e in entries:
        if e.actor in test_actors:
            test.append(e)
        elif e.actor in val_actors:
            val.append(e)
        else:
            train.append(e)
    return train, val, test


def _safe_tensor_from_array(arr: np.ndarray, dtype: 'torch.dtype') -> 'torch.Tensor':
    _require_torch()
    arr = np.nan_to_num(np.asarray(arr), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return torch.from_numpy(arr).to(dtype=dtype)
    except Exception:
        return torch.tensor(arr.tolist(), dtype=dtype)


class PilatesTemporalLifterDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[ClipEntry],
        window_size: int = 81,
        stride: int = 27,
        phase_scheme: str = "progress3",
        augment_occlusion: bool = False,
        seed: int = 42,
    ) -> None:
        _require_torch()
        self.entries = list(entries)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.phase_scheme = phase_scheme
        self.augment_occlusion = augment_occlusion
        self.rng = np.random.default_rng(seed)
        self._clip_cache: Dict[int, Dict[str, Any]] = {}
        self.windows: List[Tuple[int, int]] = []
        self.class_to_idx = self._build_class_index(self.entries)
        self.level_to_idx = self._build_level_index(self.entries)

        for idx, entry in enumerate(self.entries):
            length = self._trimmed_length(entry)
            if length <= 0:
                continue
            if length <= self.window_size:
                self.windows.append((idx, 0))
            else:
                for start in range(0, max(1, length - self.window_size + 1), self.stride):
                    self.windows.append((idx, start))
                if self.windows[-1] != (idx, length - self.window_size):
                    self.windows.append((idx, length - self.window_size))

    @staticmethod
    def _build_class_index(entries: Sequence[ClipEntry]) -> Dict[str, int]:
        names = sorted({e.category_2 for e in entries})
        return {name: i for i, name in enumerate(names)}

    @staticmethod
    def _build_level_index(entries: Sequence[ClipEntry]) -> Dict[str, int]:
        names = sorted({e.level for e in entries})
        return {name: i for i, name in enumerate(names)}

    @staticmethod
    def _trimmed_length(entry: ClipEntry) -> int:
        return max(0, int(entry.end_frame) - int(entry.start_frame) + 1)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_clip(self, idx: int) -> Dict[str, Any]:
        if idx in self._clip_cache:
            return self._clip_cache[idx]

        entry = self.entries[idx]
        x2d = read_keypoint_csv(Path(entry.csv_2d), dims=2)
        y3d = read_keypoint_csv(Path(entry.csv_3d), dims=3)

        s = max(0, int(entry.start_frame))
        e = min(len(x2d) - 1, int(entry.end_frame))
        x2d = x2d[s:e + 1]
        y3d = y3d[s:e + 1]

        x2d = normalize_skeleton(x2d)
        y3d = normalize_skeleton(y3d)

        if self.phase_scheme == "cyclic4":
            phases = cyclic_anchor_phase_labels(y3d)
            phase_names = PHASE_NAMES_4
        else:
            phases = coarse_progress_phase_labels(y3d)
            phase_names = PHASE_NAMES_3

        payload = {
            "x2d": x2d.astype(np.float32),
            "y3d": y3d.astype(np.float32),
            "phase": phases.astype(np.int64),
            "category_idx": self.class_to_idx[entry.category_2],
            "level_idx": self.level_to_idx[entry.level],
            "phase_names": phase_names,
            "meta": entry,
        }
        self._clip_cache[idx] = payload
        return payload

    @staticmethod
    def _slice_or_pad(arr: np.ndarray, start: int, size: int) -> np.ndarray:
        end = start + size
        if end <= arr.shape[0]:
            return arr[start:end]
        pad_count = end - arr.shape[0]
        pad_values = np.repeat(arr[-1:], pad_count, axis=0)
        return np.concatenate([arr[start:], pad_values], axis=0)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        _require_torch()
        clip_idx, start = self.windows[item]
        clip = self._load_clip(clip_idx)

        x2d = self._slice_or_pad(clip["x2d"], start, self.window_size)
        y3d = self._slice_or_pad(clip["y3d"], start, self.window_size)
        phase = self._slice_or_pad(clip["phase"], start, self.window_size)

        if self.augment_occlusion:
            x2d_aug, obs_mask = apply_occlusion_augmentation(x2d, self.rng)
        else:
            x2d_aug = x2d
            obs_mask = build_observation_mask(x2d)

        x = np.concatenate([x2d_aug, obs_mask], axis=-1)
        x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        y3d = np.nan_to_num(y3d.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "x": _safe_tensor_from_array(x, dtype=torch.float32),
            "y3d": _safe_tensor_from_array(y3d, dtype=torch.float32),
            "phase": _safe_tensor_from_array(phase, dtype=torch.long),
            "category": torch.tensor(clip["category_idx"], dtype=torch.long),
            "level": torch.tensor(clip["level_idx"], dtype=torch.long),
            "meta": {
                "clip_id": clip["meta"].clip_id,
                "actor": clip["meta"].actor,
                "camera": clip["meta"].camera,
                "category_2": clip["meta"].category_2,
            },
        }
