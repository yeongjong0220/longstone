"""
파이프라인 컴포넌트별 정밀 latency 측정 (각 1000회).

측정 대상:
  1. MediaPipe Pose Heavy / Lite  → 실제 frame 추론
  2. TemporalLifter forward (window=81)
  3. Landmark2DSmoother
  4. Frame3DSmoother
  5. OcclusionAwareJointBlender
  6. Bone-length lock
  7. AngleSmoother
  8. score_frame_distribution

CPU 기준 측정. GPU 사용 시 lifter 부분만 큰 차이 (CPU의 10~30배 빠름).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

LIFTER_DIR = THIS.parents[2] / "pilates_temporal_lifter"
sys.path.insert(0, str(LIFTER_DIR))


def _stats(samples_ms):
    s = sorted(samples_ms)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s), 3),
        "median_ms": round(statistics.median(s), 3),
        "p50_ms": round(s[n // 2], 3),
        "p95_ms": round(s[min(int(0.95 * n), n - 1)], 3),
        "p99_ms": round(s[min(int(0.99 * n), n - 1)], 3),
        "min_ms": round(min(s), 3),
        "max_ms": round(max(s), 3),
        "std_ms": round(statistics.stdev(s) if n > 1 else 0, 3),
    }


def bench_lifter(n_iter: int = 200) -> Dict:
    import torch
    from model import TemporalLifterConfig, TemporalLifterWithPhaseHead
    print("\n[bench] TemporalLifter (CPU)")
    ckpt_path = LIFTER_DIR / "runs" / "the_seal_progress3_angle_causal_v1" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = LIFTER_DIR / "runs" / "the_seal_progress3_lift_only_v1" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TemporalLifterConfig(**ckpt["config"])
    model = TemporalLifterWithPhaseHead(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = torch.randn(1, 81, cfg.num_joints, cfg.in_features)
    with torch.no_grad():
        for _ in range(3):
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


def bench_smoothers(n_iter: int = 1000) -> Dict:
    from core.occlusion_robust import (Frame3DSmoother, Landmark2DSmoother,
                                        OcclusionAwareJointBlender)
    from types import SimpleNamespace

    print("\n[bench] Smoothers / Blender (CPU)")
    # Landmark2D
    sm = Landmark2DSmoother()
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
    times_l = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sm(lms)
        times_l.append((time.perf_counter() - t0) * 1000.0)

    # Frame3D Smoother
    fs = Frame3DSmoother(window=7, polyorder=2)
    for _ in range(8):
        fs(np.random.randn(15, 3).astype(np.float32))
    times_f = []
    for _ in range(n_iter):
        f = np.random.randn(15, 3).astype(np.float32)
        t0 = time.perf_counter()
        fs(f)
        times_f.append((time.perf_counter() - t0) * 1000.0)

    # Occlusion blender
    ob = OcclusionAwareJointBlender()
    ob(np.random.randn(15, 3).astype(np.float32), [1.0] * 15)
    times_o = []
    for _ in range(n_iter):
        f = np.random.randn(15, 3).astype(np.float32)
        t0 = time.perf_counter()
        ob(f, [1.0] * 15)
        times_o.append((time.perf_counter() - t0) * 1000.0)

    return {
        "Landmark2DSmoother": _stats(times_l),
        "Frame3DSmoother": _stats(times_f),
        "OcclusionAwareJointBlender": _stats(times_o),
    }


def bench_bone_lock(n_iter: int = 500) -> Dict:
    from core.lifting_postprocess import enforce_bone_length
    print("\n[bench] Bone-length lock")
    seq = np.random.randn(60, 15, 3).astype(np.float32)
    for _ in range(3):
        enforce_bone_length(seq)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        enforce_bone_length(seq)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


def bench_scoring(n_iter: int = 2000) -> Dict:
    from core.distribution_scorer import load_distribution_rubrics, score_frame_distribution
    print("\n[bench] Scoring (per frame)")
    rubrics = load_distribution_rubrics(THIS.parents[1] / "reports" / "pose_stats.json")
    if not rubrics:
        return {"skipped": "pose_stats.json not found"}
    r = next(iter(rubrics.values()))
    angles = {"hip": 80.0, "knee": 36.0, "trunk": 120.0}
    score_frame_distribution(angles, r)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        score_frame_distribution(angles, r)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


def bench_mediapipe(variant: str = "heavy", n_iter: int = 100) -> Dict:
    """실제 MediaPipe로 1280×720 BGR 더미 프레임 추론 latency"""
    print(f"\n[bench] MediaPipe Pose Landmarker ({variant})")
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        print(f"  skipped: {e}")
        return {"skipped": str(e)}

    candidates = [
        THIS.parents[2] / f"pose_landmarker_{variant}.task",
        THIS.parents[1] / f"pose_landmarker_{variant}.task",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        url = (f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               f"pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task")
        target = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        model_path = target

    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    rgb = np.random.randint(80, 200, (720, 1280, 3), dtype=np.uint8)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    for _ in range(5):
        detector.detect(img)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


def main():
    out: Dict = {}
    out["lifter_cpu"] = bench_lifter()
    smoothers = bench_smoothers()
    out.update({f"smoother.{k}": v for k, v in smoothers.items()})
    out["bone_lock_60frames"] = bench_bone_lock()
    out["distribution_scorer_per_frame"] = bench_scoring()
    out["mediapipe.heavy"] = bench_mediapipe("heavy")
    out["mediapipe.lite"] = bench_mediapipe("lite")

    print("\n" + "=" * 88)
    print(f"{'Component':<48} {'mean(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10}")
    print("-" * 88)
    for k, v in out.items():
        if isinstance(v, dict) and "mean_ms" in v:
            print(f"{k:<48} {v['mean_ms']:>10} {v['p95_ms']:>10} {v['p99_ms']:>10}")
        elif isinstance(v, dict) and "skipped" in v:
            print(f"{k:<48} skipped: {v['skipped']}")
    print("=" * 88)

    out_path = THIS.parents[1] / "reports" / "latency_components.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
