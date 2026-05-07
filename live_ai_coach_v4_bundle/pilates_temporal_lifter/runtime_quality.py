from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from kinematic_scoring import (
    extract_kinematic_features,
    mahalanobis_d2,
    phase_labels_for_sequence,
    rolling_median,
    status_from_score,
)


def load_scorer(path: str | Path | None) -> Dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("model_type") != "phase_mahalanobis_kinematic_scorer":
        raise ValueError(f"Unsupported scorer model: {p}")
    return model


def score_sequence_3d(
    seq3d: np.ndarray,
    model: Dict,
    decision_window: int = 5,
) -> pd.DataFrame:
    if seq3d is None or len(seq3d) == 0:
        return pd.DataFrame()

    seq3d = np.asarray(seq3d, dtype=np.float64)
    phase_names: List[str] = list(model["phase_names"])
    phase_idx = phase_labels_for_sequence(seq3d, model["phase_scheme"])
    features = extract_kinematic_features(seq3d, model["feature_set"], int(model["smooth_kernel"]))
    x = features.to_numpy(dtype=np.float64)

    raw_scores = np.zeros(len(features), dtype=np.float64)
    for idx, phase_name in enumerate(phase_names):
        mask = phase_idx == idx
        if not np.any(mask):
            continue
        phase_model = model["phases"][phase_name]
        mean = np.asarray(phase_model["mean"], dtype=np.float64)
        inv_cov = np.asarray(phase_model["inv_cov"], dtype=np.float64)
        raw_scores[mask] = mahalanobis_d2(x[mask], mean, inv_cov)

    scores = rolling_median(raw_scores, int(decision_window))
    rows = pd.DataFrame({"frame": np.arange(len(features)), "phase_idx": phase_idx})
    rows["phase_name"] = [phase_names[int(i)] if 0 <= int(i) < len(phase_names) else "UNKNOWN" for i in phase_idx]
    for col in features.columns:
        rows[col] = features[col].to_numpy(dtype=np.float64)
    rows["mahalanobis_d2_raw"] = raw_scores
    rows["mahalanobis_d2"] = scores

    statuses = []
    top_features = []
    top_feature_z = []
    for phase_name, score, feat_row in zip(rows["phase_name"], rows["mahalanobis_d2"], x):
        if phase_name not in model["phases"]:
            statuses.append("FAIL")
            top_features.append("unknown_phase")
            top_feature_z.append(0.0)
            continue
        phase_model = model["phases"][phase_name]
        statuses.append(
            status_from_score(
                float(score),
                float(phase_model["pass_threshold"]),
                float(phase_model["warn_threshold"]),
            )
        )
        mean = np.asarray(phase_model["mean"], dtype=np.float64)
        cov = np.asarray(phase_model["cov"], dtype=np.float64)
        std = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
        z = (feat_row - mean) / std
        top_idx = int(np.argmax(np.abs(z)))
        top_features.append(model["feature_names"][top_idx])
        top_feature_z.append(float(z[top_idx]))

    rows["status"] = statuses
    rows["top_feature"] = top_features
    rows["top_feature_z"] = top_feature_z
    return rows


def summarize_score_rows(rows: pd.DataFrame) -> Dict:
    if rows is None or len(rows) == 0:
        return {
            "frames": 0,
            "status_counts": {},
            "phase_counts": {},
            "top_issue_counts": {},
            "worst_frames": [],
        }

    issue_rows = rows[rows["status"] != "PASS"].copy()
    worst_cols = ["frame", "phase_name", "status", "mahalanobis_d2", "top_feature", "top_feature_z"]
    worst_frames = (
        rows.sort_values("mahalanobis_d2", ascending=False)
        .head(8)[worst_cols]
        .to_dict(orient="records")
    )
    return {
        "frames": int(len(rows)),
        "status_counts": {k: int(v) for k, v in rows["status"].value_counts().to_dict().items()},
        "phase_counts": {k: int(v) for k, v in rows["phase_name"].value_counts().to_dict().items()},
        "top_issue_counts": {k: int(v) for k, v in issue_rows["top_feature"].value_counts().to_dict().items()},
        "mean_mahalanobis_d2": float(rows["mahalanobis_d2"].mean()),
        "max_mahalanobis_d2": float(rows["mahalanobis_d2"].max()),
        "warn_or_fail_ratio": float((rows["status"] != "PASS").mean()),
        "worst_frames": worst_frames,
    }


def format_summary_for_prompt(pose_name: str, summary: Dict) -> str:
    status_counts = summary.get("status_counts", {})
    top_issues = summary.get("top_issue_counts", {})
    worst = summary.get("worst_frames", [])[:5]
    return (
        f"Pose: {pose_name}\n"
        f"Frames analyzed: {summary.get('frames', 0)}\n"
        f"PASS/WARN/FAIL counts: {status_counts}\n"
        f"Top issue features: {top_issues}\n"
        f"Mean Mahalanobis d2: {summary.get('mean_mahalanobis_d2', 0):.3f}\n"
        f"Max Mahalanobis d2: {summary.get('max_mahalanobis_d2', 0):.3f}\n"
        f"Worst frames: {worst}"
    )
