"""
TemporalLifterWithPhaseHead → ONNX 변환 + 검증 스크립트.

모바일/임베디드 포팅을 위한 Phase 2 단계 (MOBILE_DEPLOYMENT.md 참고).

사용:
  python eval/export_lifter_onnx.py \
    --ckpt ../pilates_temporal_lifter/runs/the_seal_progress3_angle_causal_v1/best.pt \
    --out reports/lifter_causal.onnx \
    --verify

옵션:
  --quantize    : INT8 동적 양자화도 함께 수행 → reports/lifter_int8.onnx
  --benchmark   : 추론 latency 비교 (PyTorch vs ONNX vs ONNX-INT8)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

# pilates_temporal_lifter 임포트 경로
LIFTER_DIR = THIS.parents[2] / "pilates_temporal_lifter"
sys.path.insert(0, str(LIFTER_DIR))


def export(ckpt_path: Path, out_path: Path, window: int = 81) -> Path:
    import torch
    from model import TemporalLifterConfig, TemporalLifterWithPhaseHead

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TemporalLifterConfig(**ckpt["config"])
    model = TemporalLifterWithPhaseHead(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[cfg] num_joints={cfg.num_joints}, hidden={cfg.hidden_dim}, "
          f"dilations={cfg.dilations}, causal={cfg.causal}")

    dummy = torch.randn(1, window, cfg.num_joints, cfg.in_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["x"], output_names=["pose3d", "phase_logits"],
        dynamic_axes={"x": {0: "batch", 1: "time"},
                       "pose3d": {0: "batch", 1: "time"},
                       "phase_logits": {0: "batch", 1: "time"}},
        opset_version=17,
    )
    main_mb = out_path.stat().st_size / (1024 * 1024)
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    data_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0.0
    total_mb = main_mb + data_mb
    print(f"[ok] saved {out_path}")
    print(f"     graph   : {main_mb:.2f} MB")
    if data_mb > 0:
        print(f"     weights : {data_mb:.2f} MB  ({data_path.name})")
    print(f"     total   : {total_mb:.2f} MB")
    return out_path


def verify(ckpt_path: Path, onnx_path: Path, window: int = 81) -> dict:
    """PyTorch vs ONNX 출력 비교"""
    import torch
    from model import TemporalLifterConfig, TemporalLifterWithPhaseHead

    print(f"\n[verify] PyTorch vs ONNX numerical match")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TemporalLifterConfig(**ckpt["config"])
    model = TemporalLifterWithPhaseHead(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    np.random.seed(42)
    x = np.random.randn(1, window, cfg.num_joints, cfg.in_features).astype(np.float32)

    with torch.no_grad():
        t_pose, t_phase = model(torch.from_numpy(x))
    t_pose = t_pose.numpy()
    t_phase = t_phase.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_pose, o_phase = sess.run(None, {"x": x})

    pose_diff = float(np.max(np.abs(t_pose - o_pose)))
    phase_diff = float(np.max(np.abs(t_phase - o_phase)))
    print(f"  pose3d max-abs-diff : {pose_diff:.6e}")
    print(f"  phase   max-abs-diff: {phase_diff:.6e}")
    ok = pose_diff < 1e-4 and phase_diff < 1e-4
    print(f"  match: {'OK' if ok else 'WARN (diff > 1e-4)'}")
    return {"pose_max_diff": pose_diff, "phase_max_diff": phase_diff, "match": ok}


def quantize_dynamic(in_path: Path) -> Path:
    """ONNX dynamic INT8 quantization"""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out_path = in_path.with_name(in_path.stem + "_int8.onnx")
    print(f"\n[quantize] INT8 dynamic -> {out_path}")
    quantize_dynamic(str(in_path), str(out_path), weight_type=QuantType.QUInt8)
    # 정확한 총 크기 (external data 포함)
    sz_in_main = in_path.stat().st_size / (1024 * 1024)
    sz_in_data = 0.0
    data_p = in_path.with_suffix(in_path.suffix + ".data")
    if data_p.exists():
        sz_in_data = data_p.stat().st_size / (1024 * 1024)
    sz_in_total = sz_in_main + sz_in_data
    sz_out = out_path.stat().st_size / (1024 * 1024)
    print(f"  before: {sz_in_total:.2f} MB (graph {sz_in_main:.2f} + weights {sz_in_data:.2f})")
    print(f"  after : {sz_out:.2f} MB")
    if sz_in_total > 0:
        print(f"  reduction: {100*(sz_in_total - sz_out)/sz_in_total:.1f}% smaller  ({sz_in_total/sz_out:.2f}x)")
    return out_path


def benchmark(ckpt_path: Path, onnx_path: Path, int8_path: Path | None = None,
              window: int = 81, n_iter: int = 50) -> dict:
    """추론 latency 비교: PyTorch CPU vs ONNX Runtime vs ONNX INT8"""
    import torch
    from model import TemporalLifterConfig, TemporalLifterWithPhaseHead

    print(f"\n[bench] inference latency (mean over {n_iter} iters)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TemporalLifterConfig(**ckpt["config"])
    model = TemporalLifterWithPhaseHead(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_np = np.random.randn(1, window, cfg.num_joints, cfg.in_features).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(x_t)

    # PyTorch
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(x_t)
    pt_ms = (time.perf_counter() - t0) / n_iter * 1000.0
    print(f"  PyTorch CPU      : {pt_ms:6.2f} ms")

    # ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for _ in range(3):
        sess.run(None, {"x": x_np})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sess.run(None, {"x": x_np})
    onnx_ms = (time.perf_counter() - t0) / n_iter * 1000.0
    print(f"  ONNX Runtime CPU : {onnx_ms:6.2f} ms  ({pt_ms/onnx_ms:.2f}x speedup)")

    int8_ms = None
    if int8_path and int8_path.exists():
        sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        for _ in range(3):
            sess8.run(None, {"x": x_np})
        t0 = time.perf_counter()
        for _ in range(n_iter):
            sess8.run(None, {"x": x_np})
        int8_ms = (time.perf_counter() - t0) / n_iter * 1000.0
        print(f"  ONNX INT8        : {int8_ms:6.2f} ms  ({pt_ms/int8_ms:.2f}x speedup)")

    return {"pytorch_ms": pt_ms, "onnx_ms": onnx_ms, "onnx_int8_ms": int8_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="ONNX 출력 경로 (기본: reports/lifter_<ckpt_stem>.onnx)")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--verify", action="store_true", help="PyTorch vs ONNX 출력 일치 검증")
    ap.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization 수행")
    ap.add_argument("--benchmark", action="store_true", help="추론 속도 비교")
    args = ap.parse_args()

    out_path = args.out or (THIS.parents[1] / "reports" / f"lifter_{args.ckpt.parent.name}.onnx")

    try:
        export(args.ckpt, out_path, args.window)
    except ImportError as e:
        print(f"[error] {e}\n  필요: pip install torch onnx onnxruntime")
        return 1

    if args.verify:
        try:
            verify(args.ckpt, out_path, args.window)
        except ImportError as e:
            print(f"[warn] verify skipped: {e}")

    int8_path = None
    if args.quantize:
        try:
            int8_path = quantize_dynamic(out_path)
        except ImportError as e:
            print(f"[warn] quantize skipped: {e}")

    if args.benchmark:
        try:
            benchmark(args.ckpt, out_path, int8_path, args.window)
        except ImportError as e:
            print(f"[warn] benchmark skipped: {e}")

    print(f"\n[done] ONNX artifacts ready in: {out_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
