"""
Paper-style OnPose v6 system architecture diagram.
All English labels for clean rendering across systems.
Outputs: reports/architecture_paper.png (300dpi)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

THIS = Path(__file__).resolve()
OUT = THIS.parents[1] / "reports" / "architecture_paper.png"

C_INPUT = "#E8F4FA";   C_INPUT_E = "#5BA3D0"
C_BACK = "#FFF4D6";    C_BACK_E = "#E5A93C"
C_HEAD = "#E8F8E0";    C_HEAD_E = "#4FA940"
C_POST = "#F5E8FA";    C_POST_E = "#9C5BC4"
C_SCORE = "#FAE0E0";   C_SCORE_E = "#C84A4A"
C_UI = "#E0E8F0";      C_UI_E = "#5A6F8C"
C_TEXT = "#1A1A1A";    C_ARROW = "#444444";   C_SUB = "#666666"


def box(ax, xy, w, h, title, fill, edge, sub=None, fs=10, sub_fs=8):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                       facecolor=fill, edgecolor=edge, linewidth=1.7)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (0.10 if sub else 0), title,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=C_TEXT)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.14, sub, ha="center", va="center",
                fontsize=sub_fs, color=C_SUB)


def arrow(ax, p1, p2, label=None, curve=0.0, label_off=(0, 0)):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                        color=C_ARROW, linewidth=1.4,
                        connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_off[0]
        my = (p1[1] + p2[1]) / 2 + label_off[1]
        ax.text(mx, my, label, fontsize=8, color=C_SUB, ha="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.88))


def render():
    fig, ax = plt.subplots(figsize=(16.5, 9.5), dpi=150)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.965, "OnPose v6 — On-device Real-time Posture Coaching",
             ha="center", fontsize=16, fontweight="bold", color="#0F1F38")
    fig.text(0.5, 0.935,
             "MediaPipe (33 lm) → Lifter (TCN 1.6M params) → "
             "Occlusion-robust postprocess → Distribution scoring → Friendly LLM",
             ha="center", fontsize=10, color="#3F5470", style="italic")

    # ===== Row 1: Input → MediaPipe → Smoother =====
    y = 7.7
    box(ax, (0.3, y), 1.8, 1.0, "Webcam",
        C_INPUT, C_INPUT_E, sub="1280x720 BGR\n~30 fps")
    box(ax, (2.8, y), 2.6, 1.0, "MediaPipe Pose",
        C_BACK, C_BACK_E,
        sub="Heavy: 26M par, 1.05G FLOPs\nLite : 2M par,  85M FLOPs")
    box(ax, (6.1, y), 2.6, 1.0, "Landmark2D\nSmoother",
        C_POST, C_POST_E,
        sub="EMA, vis-aware hold\n0.03 ms / frame")

    arrow(ax, (2.1, y + 0.5), (2.8, y + 0.5))
    arrow(ax, (5.4, y + 0.5), (6.1, y + 0.5), label="33 lm + vis")

    # ===== Row 2: TemporalLifter =====
    y2 = 5.6
    cont = FancyBboxPatch((2.8, y2 - 0.15), 11.0, 1.7,
                          boxstyle="round,pad=0.05,rounding_size=0.18",
                          facecolor=C_BACK, edgecolor=C_BACK_E,
                          linewidth=2.2, alpha=0.45)
    ax.add_patch(cont)
    ax.text(2.9, y2 + 1.3,
            "TemporalLifterWithPhaseHead (TCN, hidden=256, dilations=[1,2,4,8])  "
            "—  1.603M params, 130M FLOPs / window",
            fontsize=10.5, fontweight="bold", color="#7E5A1E")

    box(ax, (2.95, y2), 1.7, 1.0, "Sliding\nWindow",
        "#FFFCEC", "#D9A93C", sub="(B, T=81, J=15, C=3)\nC=(x,y,obs_mask)", fs=9, sub_fs=7.5)
    box(ax, (4.85, y2), 1.3, 1.0, "Input\nProj 1x1",
        "#FFFCEC", "#D9A93C", sub="45 -> 256\n0.9M FLOPs", fs=9, sub_fs=7.5)
    # 4 blocks
    for i, d in enumerate([1, 2, 4, 8]):
        x_b = 6.35 + i * 1.15
        box(ax, (x_b, y2), 1.0, 1.0, f"Res-Conv\nd={d}",
            "#FFE9B0", "#C68A1F", sub="Causal\nBN+GELU\n32M", fs=8.5, sub_fs=6.5)
    # heads
    box(ax, (11.5, y2 + 0.55), 1.4, 0.45, "Pose Head",
        C_HEAD, C_HEAD_E, sub="-> (T, 15, 3) 3D", fs=9, sub_fs=7)
    box(ax, (11.5, y2 + 0.0), 1.4, 0.45, "Phase Head",
        C_HEAD, C_HEAD_E, sub="-> (T, 3) logits", fs=9, sub_fs=7)

    arrow(ax, (8.7, y + 0.5), (3.8, y2 + 1.0),
          label="MP-33 -> Lifter-15 map", curve=-0.18, label_off=(0, 0.1))

    # ===== Row 3: postprocessing chain =====
    y3 = 3.5
    box(ax, (0.3, y3), 1.9, 1.0, "Frame3D\nSmoother",
        C_POST, C_POST_E, sub="Savgol w=7\n10.8 ms")
    box(ax, (2.5, y3), 2.1, 1.0, "Occlusion-aware\nJoint Blender",
        C_POST, C_POST_E, sub="vis-weighted hold\n0.002 ms")
    box(ax, (4.9, y3), 1.7, 1.0, "Bone-length\nLock",
        C_POST, C_POST_E, sub="anchor=torso\n6 ms / 60f")
    box(ax, (6.9, y3), 2.0, 1.0, "3D Angle Calc\nhip / knee / trunk",
        "#EEE8FA", "#9C5BC4", sub="vis-weighted L/R")
    box(ax, (9.2, y3), 1.7, 1.0, "Angle EMA\n+ Outlier Clip",
        C_POST, C_POST_E, sub="alpha=0.45\n|dx|<12 deg/frame")

    arrow(ax, (12.9, y2 + 0.55), (1.25, y3 + 1.0),
          label="(T, 15, 3)", curve=-0.45, label_off=(0, 0.18))
    arrow(ax, (2.2, y3 + 0.5), (2.5, y3 + 0.5))
    arrow(ax, (4.6, y3 + 0.5), (4.9, y3 + 0.5))
    arrow(ax, (6.6, y3 + 0.5), (6.9, y3 + 0.5))
    arrow(ax, (8.9, y3 + 0.5), (9.2, y3 + 0.5))

    # ===== Row 4: Scoring + UI =====
    y4 = 1.4
    box(ax, (0.3, y4), 2.7, 1.2, "Distribution\nScorer",
        C_SCORE, C_SCORE_E,
        sub="z-score(mu, sigma)\n+ velocity likelihood\n0.007 ms")
    box(ax, (3.3, y4), 2.8, 1.2, "Friendly LLM Coach",
        C_SCORE, C_SCORE_E,
        sub="Gemini 2.5 Flash  /\nOffline template\n0.8-2.5 s  /  <1 ms")
    box(ax, (6.4, y4), 2.7, 1.2, "Session Aggregator",
        C_SCORE, C_SCORE_E,
        sub="5s capture -> score\nO/X accuracy %\nbest-shot save")
    box(ax, (9.4, y4), 6.0, 1.2, "Light-themed UI  (1280 x 800)",
        C_UI, C_UI_E,
        sub="cam + skeleton overlay   |   expert video card   |   live accuracy\n"
            "angle dials   |   friendly coach   |   sound/voice   |   record mp4")

    arrow(ax, (9.85, y3 + 0.55), (1.65, y4 + 1.2), label="angles", curve=-0.40,
          label_off=(0, 0.15))
    arrow(ax, (3.0, y4 + 0.6), (3.3, y4 + 0.6))
    arrow(ax, (6.1, y4 + 0.6), (6.4, y4 + 0.6))
    arrow(ax, (9.1, y4 + 0.6), (9.4, y4 + 0.6))

    # sidebar: calibrated rubric
    box(ax, (13.2, y3 - 0.1), 3.0, 1.2,
        "Calibrated Rubric",
        "#F4F4F4", "#999999",
        sub="AI Hub 216 - 44 actors\nper-angle mu, sigma\nvelocity p50 / p95")
    arrow(ax, (14.7, y3 - 0.1), (1.65, y4 + 1.2),
          label="pose_stats.json", curve=-0.45, label_off=(0, -0.20))

    # ===== Bottom info =====
    fig.text(0.02, 0.045,
             "Runtime modes:  STANDARD (Heavy, ~12-15 fps)  |  "
             "LITE (Lite + Lifter off, ~40-55 fps mobile)  |  "
             "ONNX INT8 = 1.59 MB  (3.87x smaller, 1.78x faster on CPU)",
             fontsize=9, color="#3F5470")
    fig.text(0.02, 0.024,
             "Accuracy:  Hip PCK@15deg = 89.86% raw / 95.65% with GT smoothing   "
             "|   Hip PCK@20deg + bone-lock = 94.20%   "
             "|   End-phase verdict agreement = 100%",
             fontsize=9, color="#4F6B40", fontweight="bold")

    # legend
    legend = [
        mpatches.Patch(facecolor=C_INPUT, edgecolor=C_INPUT_E, label="Input"),
        mpatches.Patch(facecolor=C_BACK, edgecolor=C_BACK_E, label="Backbone / Lifter"),
        mpatches.Patch(facecolor=C_HEAD, edgecolor=C_HEAD_E, label="Heads"),
        mpatches.Patch(facecolor=C_POST, edgecolor=C_POST_E, label="Postprocessing"),
        mpatches.Patch(facecolor=C_SCORE, edgecolor=C_SCORE_E, label="Scoring / LLM"),
        mpatches.Patch(facecolor=C_UI, edgecolor=C_UI_E, label="UI"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.000))

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    render()
