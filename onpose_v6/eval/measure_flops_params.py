"""
TemporalLifterWithPhaseHead + 서브 모듈별 FLOPs / Params 측정.

- thop.profile 사용 (PyTorch ops 카운트)
- 각 ResidualTemporalBlock 별로도 측정
- MediaPipe Pose Landmarker는 자체 측정 불가 (TFLite 블랙박스) → 공식 발표값 사용
- 룰베이스 (angle_scorer, distribution_scorer)는 FLOPs 무시 가능 수준 (산술만)

사용:
  python eval/measure_flops_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

LIFTER_DIR = THIS.parents[2] / "pilates_temporal_lifter"
sys.path.insert(0, str(LIFTER_DIR))


def measure_lifter() -> dict:
    import torch
    from thop import profile, clever_format
    from model import TemporalLifterConfig, TemporalLifterWithPhaseHead

    # Causal lifter 체크포인트 로드
    ckpt_path = LIFTER_DIR / "runs" / "the_seal_progress3_angle_causal_v1" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = LIFTER_DIR / "runs" / "the_seal_progress3_lift_only_v1" / "best.pt"
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TemporalLifterConfig(**ckpt["config"])
    model = TemporalLifterWithPhaseHead(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 전체 모델
    x = torch.randn(1, 81, cfg.num_joints, cfg.in_features)
    flops_total, params_total = profile(model, inputs=(x,), verbose=False)

    # 컴포넌트별 측정
    # Input projection 단독
    # forward 안에서 reshape 처리하므로 시뮬 입력 직접 만듦
    proj_in = torch.randn(1, cfg.num_joints * cfg.in_features, 81)
    flops_proj, params_proj = profile(model.input_proj, inputs=(proj_in,), verbose=False)

    # 한 block
    block_in = torch.randn(1, cfg.hidden_dim, 81)
    one_block = model.blocks[0]
    flops_block, params_block = profile(one_block, inputs=(block_in,), verbose=False)

    # 두 head
    head_in = torch.randn(1, cfg.hidden_dim, 81)
    flops_pose, params_pose = profile(model.pose_head, inputs=(head_in,), verbose=False)
    flops_phase, params_phase = profile(model.phase_head, inputs=(head_in,), verbose=False)

    def fmt(flops, params):
        f, p = clever_format([flops, params], "%.3f")
        return f, p

    print("\n=== Lifter Component Breakdown ===")
    print(f"{'Component':<30} {'FLOPs':>12} {'Params':>12}")
    print("-" * 60)
    components = [
        ("Input Projection (Conv1d 45→256)", flops_proj, params_proj),
        ("ResidualTemporalBlock (×4)", flops_block * 4, params_block * 4),
        (f"  one block (d=1, hidden={cfg.hidden_dim})", flops_block, params_block),
        ("Pose Head (Conv1d 256→45)", flops_pose, params_pose),
        ("Phase Head (Conv1d 256→3)", flops_phase, params_phase),
        ("TOTAL (forward 1×81 frames)", flops_total, params_total),
    ]
    for name, f, p in components:
        f_s, p_s = fmt(f, p)
        print(f"{name:<30} {f_s:>12} {p_s:>12}")

    return {
        "config": {
            "num_joints": cfg.num_joints,
            "in_features": cfg.in_features,
            "hidden_dim": cfg.hidden_dim,
            "dilations": cfg.dilations,
            "kernel_size": cfg.kernel_size,
            "causal": cfg.causal,
            "num_phases": cfg.num_phases,
        },
        "input_shape": "(1, 81, 15, 3)",
        "flops": {
            "input_projection": int(flops_proj),
            "single_block": int(flops_block),
            "all_4_blocks": int(flops_block * 4),
            "pose_head": int(flops_pose),
            "phase_head": int(flops_phase),
            "total": int(flops_total),
        },
        "params": {
            "input_projection": int(params_proj),
            "single_block": int(params_block),
            "all_4_blocks": int(params_block * 4),
            "pose_head": int(params_pose),
            "phase_head": int(params_phase),
            "total": int(params_total),
        },
    }


def measure_mediapipe_estimated() -> dict:
    """MediaPipe Pose Landmarker는 TFLite 블랙박스라 직접 측정 불가.
    Google 공식 발표값 사용."""
    return {
        "lite":  {"params": "~2.0M",  "flops": "~85M  (per frame)", "size_mb": 3.2},
        "full":  {"params": "~7.5M",  "flops": "~285M (per frame)", "size_mb": 9.0},
        "heavy": {"params": "~26.0M", "flops": "~1.05G (per frame)", "size_mb": 30.0},
    }


def main():
    lifter = measure_lifter()
    mp_est = measure_mediapipe_estimated()

    # 종합
    print("\n=== System-wide Estimate (per frame) ===")
    print(f"{'Pipeline (Heavy mode)':<40} {'Params':>10} {'FLOPs':>14}")
    print("-" * 70)
    print(f"{'MediaPipe Pose Heavy':<40} {'~26.0M':>10} {'~1.05G':>14}")
    print(f"{'TemporalLifter (forward window=81)':<40} "
          f"{lifter['params']['total']/1e6:>9.3f}M  {lifter['flops']['total']/1e9:>11.3f}G")
    print(f"{'Bone-length lock / smoothing':<40} {'<1K':>10} {'~10K':>14}")
    print(f"{'Angle calc + Distribution scorer':<40} {'-':>10} {'~1K':>14}")
    print("-" * 70)

    # 출력 저장
    out = {
        "lifter": lifter,
        "mediapipe_estimated": mp_est,
        "post_processing": {
            "bone_lock": "O(J×F) — JointInverse-Kinematics-like, ~5K FLOPs per frame",
            "frame3d_smooth": "Savgol filter (window=7) — ~3K FLOPs per frame",
            "occlusion_blend": "per-joint blend — O(J)",
        },
        "scoring": {
            "angle_scorer": "weighted average over J angles, ~50 FLOPs per frame",
            "distribution_scorer": "z-score + Gaussian likelihood — ~150 FLOPs per frame",
        },
        "system_total_per_frame": {
            "params_M": (26.0 + lifter["params"]["total"] / 1e6),
            "flops_G": (1.05 + lifter["flops"]["total"] / 1e9),
            "note": "MediaPipe Heavy + TemporalLifter forward (window=81)",
        },
    }
    out_path = THIS.parents[1] / "reports" / "flops_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
