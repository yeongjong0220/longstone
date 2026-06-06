
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SymmetricConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        conv_cls = CausalConv1d if causal else SymmetricConv1d
        self.conv1 = conv_cls(channels, kernel_size, dilation)
        self.conv2 = conv_cls(channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.drop(out)

        out = out + residual
        out = F.gelu(out)
        return out


@dataclass
class TemporalLifterConfig:
    num_joints: int = 15
    in_features: int = 3     # x, y, observed_mask
    hidden_dim: int = 256
    kernel_size: int = 3
    dilations: List[int] = None
    causal: bool = True
    dropout: float = 0.10
    num_phases: int = 3

    def __post_init__(self) -> None:
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8]


class TemporalLifterWithPhaseHead(nn.Module):
    def __init__(self, cfg: TemporalLifterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.num_joints * cfg.in_features

        self.input_proj = nn.Conv1d(input_dim, cfg.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=cfg.hidden_dim,
                    kernel_size=cfg.kernel_size,
                    dilation=d,
                    causal=cfg.causal,
                    dropout=cfg.dropout,
                )
                for d in cfg.dilations
            ]
        )
        self.pose_head = nn.Conv1d(cfg.hidden_dim, cfg.num_joints * 3, kernel_size=1)
        self.phase_head = nn.Conv1d(cfg.hidden_dim, cfg.num_phases, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, J, C)
        returns:
          pred_3d: (B, T, J, 3)
          phase_logits: (B, T, P)
        """
        b, t, j, c = x.shape
        x = x.reshape(b, t, j * c).transpose(1, 2)   # (B, JC, T)
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)

        pred_3d = self.pose_head(feat).transpose(1, 2).reshape(b, t, j, 3)
        phase_logits = self.phase_head(feat).transpose(1, 2)  # (B, T, P)
        return pred_3d, phase_logits
