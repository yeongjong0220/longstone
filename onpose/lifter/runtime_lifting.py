from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataset import JOINT_ORDER, build_observation_mask, normalize_skeleton
from model import TemporalLifterConfig, TemporalLifterWithPhaseHead


MP_POSE_MAP = {
    "Head": [0],
    "Neck": [11, 12],
    "LShoulder": [11],
    "RShoulder": [12],
    "LElbow": [13],
    "RElbow": [14],
    "LWrist": [15],
    "RWrist": [16],
    "LHip": [23],
    "RHip": [24],
    "LKnee": [25],
    "Rknee": [26],
    "LAnkle": [27],
    "RAnkle": [28],
    "Hip": [23, 24],
}


def mediapipe_landmarks_to_lifter_2d(landmarks, min_visibility: float = 0.35) -> np.ndarray:
    frame = np.zeros((len(JOINT_ORDER), 2), dtype=np.float32)
    if landmarks is None:
        return frame
    for out_idx, name in enumerate(JOINT_ORDER):
        src_ids = MP_POSE_MAP[name]
        pts = []
        for src_idx in src_ids:
            lm = landmarks[src_idx]
            if getattr(lm, "visibility", 1.0) < min_visibility:
                continue
            pts.append([float(lm.x), float(lm.y)])
        if pts:
            frame[out_idx] = np.asarray(pts, dtype=np.float32).mean(axis=0)
    return frame


class OnlineTemporalLifter:
    def __init__(
        self,
        checkpoint: str | Path,
        window_size: int = 81,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.window_size = int(window_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        cfg = TemporalLifterConfig(**ckpt["config"])
        self.model = TemporalLifterWithPhaseHead(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.phase_names = ckpt.get("phase_names", [])
        self.frames_2d: List[np.ndarray] = []

    def reset(self) -> None:
        self.frames_2d.clear()

    def append_landmarks(self, landmarks) -> Tuple[np.ndarray | None, int]:
        frame_2d = mediapipe_landmarks_to_lifter_2d(landmarks)
        self.frames_2d.append(frame_2d)
        if len(self.frames_2d) > self.window_size:
            self.frames_2d = self.frames_2d[-self.window_size :]
        return self.predict_latest()

    @torch.no_grad()
    def predict_latest(self) -> Tuple[np.ndarray | None, int]:
        if not self.frames_2d:
            return None, -1
        arr = np.asarray(self.frames_2d, dtype=np.float32)
        arr = normalize_skeleton(arr)
        obs = build_observation_mask(arr)
        x = np.concatenate([arr, obs], axis=-1)

        if len(x) < self.window_size:
            pad_count = self.window_size - len(x)
            pad = np.repeat(x[:1], pad_count, axis=0)
            x = np.concatenate([pad, x], axis=0)

        inp = torch.tensor(x.tolist(), dtype=torch.float32).unsqueeze(0).to(self.device)
        pred3d, phase_logits = self.model(inp)
        latest_3d = pred3d[0, -1].detach().cpu().numpy().astype(np.float32)
        latest_phase = int(phase_logits[0, -1].argmax().item())
        return latest_3d, latest_phase
