from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dataset import BONES, JOINT_ORDER

_IDX = {name: i for i, name in enumerate(JOINT_ORDER)}


def _bone_lengths(x: torch.Tensor) -> torch.Tensor:
    vals = []
    for a, b in BONES:
        ia, ib = _IDX[a], _IDX[b]
        vals.append(torch.linalg.norm(x[:, :, ia, :] - x[:, :, ib, :], dim=-1, keepdim=True))
    return torch.cat(vals, dim=-1)


def _angle_from_triplet(x: torch.Tensor, a: str, b: str, c: str) -> torch.Tensor:
    ia, ib, ic = _IDX[a], _IDX[b], _IDX[c]
    u = x[:, :, ia, :] - x[:, :, ib, :]
    v = x[:, :, ic, :] - x[:, :, ib, :]
    un = F.normalize(u, dim=-1, eps=1e-6)
    vn = F.normalize(v, dim=-1, eps=1e-6)
    cross = torch.linalg.norm(torch.cross(un, vn, dim=-1), dim=-1)
    dot = (un * vn).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.atan2(cross, dot)
    return torch.nan_to_num(ang, nan=0.0, posinf=3.14159265, neginf=0.0)


def _angles(x: torch.Tensor) -> torch.Tensor:
    hip_l = _angle_from_triplet(x, 'Neck', 'LHip', 'LKnee')
    hip_r = _angle_from_triplet(x, 'Neck', 'RHip', 'Rknee')
    knee_l = _angle_from_triplet(x, 'LHip', 'LKnee', 'LAnkle')
    knee_r = _angle_from_triplet(x, 'RHip', 'Rknee', 'RAnkle')
    trunk = _angle_from_triplet(x, 'Hip', 'Neck', 'Head')
    return torch.stack([hip_l, hip_r, knee_l, knee_r, trunk], dim=-1)


def compute_losses(
    pred_3d: torch.Tensor,
    y3d: torch.Tensor,
    phase_logits: torch.Tensor,
    phase: torch.Tensor,
    w_pose: float = 1.0,
    w_phase: float = 0.2,
    w_bone: float = 0.2,
    w_vel: float = 0.1,
    w_angle: float = 0.1,
) -> Dict[str, torch.Tensor]:
    pred_3d = torch.nan_to_num(pred_3d, nan=0.0, posinf=1e4, neginf=-1e4)
    y3d = torch.nan_to_num(y3d, nan=0.0, posinf=1e4, neginf=-1e4)
    phase_logits = torch.nan_to_num(phase_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    pose = F.smooth_l1_loss(pred_3d, y3d)
    phase_loss = F.cross_entropy(phase_logits.reshape(-1, phase_logits.shape[-1]), phase.reshape(-1))

    pred_bone = _bone_lengths(pred_3d)
    true_bone = _bone_lengths(y3d)
    bone = F.l1_loss(pred_bone, true_bone)

    if pred_3d.shape[1] > 1:
        pred_vel = pred_3d[:, 1:] - pred_3d[:, :-1]
        true_vel = y3d[:, 1:] - y3d[:, :-1]
        vel = F.l1_loss(pred_vel, true_vel)
    else:
        vel = pose.new_zeros(())

    angle = F.l1_loss(_angles(pred_3d), _angles(y3d))

    total = w_pose * pose + w_phase * phase_loss + w_bone * bone + w_vel * vel + w_angle * angle
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=-1e4)
    mpjpe = torch.linalg.norm(pred_3d - y3d, dim=-1).mean()
    mpjpe = torch.nan_to_num(mpjpe, nan=0.0, posinf=1e4, neginf=-1e4)

    return {
        'total': total,
        'pose': pose,
        'phase': phase_loss,
        'bone': bone,
        'vel': vel,
        'angle': angle,
        'mpjpe': mpjpe,
    }
