"""
컴포넌트별 latency 측정 — 멘토링 피드백:
"각각의 컴포넌트에서 랩타임이 얼마나 걸리는지 체크해야 한다.
백엔드에서 얼마나 걸리는지 예측이 되어야 프론트에서 사용자에게 보여줄 수 있다."

사용 예:
    profiler = LatencyProfiler()
    with profiler.measure("mediapipe"):
        ... mediapipe.detect ...
    with profiler.measure("lifter"):
        ... lifter.forward ...
    profiler.report()        # 콘솔 출력
    profiler.to_dict()       # JSON 저장용
"""
from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List


class LatencyProfiler:
    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.samples.setdefault(name, []).append(elapsed_ms)

    def record(self, name: str, elapsed_ms: float) -> None:
        self.samples.setdefault(name, []).append(elapsed_ms)

    def stats(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, vals in self.samples.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            out[name] = {
                "n": n,
                "mean_ms": round(statistics.mean(vals), 2),
                "median_ms": round(statistics.median(vals), 2),
                "p95_ms": round(sorted_vals[min(int(0.95 * n), n - 1)], 2),
                "p99_ms": round(sorted_vals[min(int(0.99 * n), n - 1)], 2),
                "max_ms": round(max(vals), 2),
                "min_ms": round(min(vals), 2),
            }
        return out

    def report(self, title: str = "Latency Report") -> str:
        """콘솔 출력용 표 + 가장 느린 컴포넌트 표시"""
        stats = self.stats()
        if not stats:
            return f"[{title}] no samples"
        lines = []
        lines.append("=" * 72)
        lines.append(f"  {title}")
        lines.append("=" * 72)
        lines.append(f"{'Component':<22} {'n':>6} {'mean(ms)':>12} {'p95(ms)':>12} {'max(ms)':>12}")
        lines.append("-" * 72)
        # mean 기준 내림차순
        ordered = sorted(stats.items(), key=lambda kv: -kv[1]["mean_ms"])
        for name, s in ordered:
            lines.append(f"{name:<22} {s['n']:>6} {s['mean_ms']:>12} {s['p95_ms']:>12} {s['max_ms']:>12}")
        lines.append("-" * 72)
        # 가장 느린 컴포넌트 (병목)
        slowest = ordered[0]
        lines.append(f"  병목 컴포넌트: {slowest[0]}  (mean {slowest[1]['mean_ms']} ms)")
        lines.append("=" * 72)
        report = "\n".join(lines)
        print(report)
        return report

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict:
        return {
            "summary": self.stats(),
            "samples_count": {k: len(v) for k, v in self.samples.items()},
        }

    def fps_estimate(self, frame_pipeline_keys: List[str] = None) -> float:
        """프레임 파이프라인 latency의 합으로 추정한 최대 FPS"""
        if frame_pipeline_keys is None:
            frame_pipeline_keys = ["frame_total"]
        stats = self.stats()
        total = 0.0
        for k in frame_pipeline_keys:
            if k in stats:
                total += stats[k]["mean_ms"]
        return 1000.0 / total if total > 0 else 0.0

    def reset(self) -> None:
        self.samples.clear()
