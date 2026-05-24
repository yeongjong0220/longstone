"""
TTA(Test-Time Augmentation) 효과를 정량 평가.

원본 pred_csv + 좌우 반전 pred_csv (별도 추론 필요)를 받아 평균.
다만 이 스크립트는 사후 시뮬레이션 — 실제로 두 번 추론한 CSV가 없으면,
GT를 좌우 반전한 뒤 다시 좌우 반전해서 노이즈를 모방하는 식으로 효과 추정.

가장 현실적인 사용:
  - 실제 라이브 추론 코드에 TTA를 켜고 끄고 비교
  - 이 스크립트는 후처리 + TTA 결합 효과 비교
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from core.lifting_postprocess import (df_to_seq, seq_to_df, enforce_bone_length,
                                        temporal_smooth_3d, postprocess_pipeline)
from core.tta import flip_skeleton_lr
from eval.lifting_accuracy import evaluate
from eval.compare_postprocess import evaluate_seq, headline, print_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--pred-csv-flipped", type=Path, default=None,
                    help="좌우 반전 입력으로 추론한 결과 CSV (별도 추론 필요). 없으면 'self-flip'으로 시뮬레이션")
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--pose", type=str, default="the_seal")
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    pred_seq = df_to_seq(pred_df)

    baseline = evaluate(args.pred_csv, args.gt_csv, pose_key=args.pose)

    # 실제 TTA: 좌우 반전 입력으로 별도 추론한 결과가 있을 때
    if args.pred_csv_flipped is not None and args.pred_csv_flipped.exists():
        flipped_df = pd.read_csv(args.pred_csv_flipped)
        flipped_seq = df_to_seq(flipped_df)
        # 반전 입력의 출력을 다시 좌우 반전해 같은 좌표계로
        unflipped = flip_skeleton_lr(flipped_seq)
        # 시퀀스 길이 맞춤
        n = min(len(pred_seq), len(unflipped))
        tta_avg = 0.5 * pred_seq[:n] + 0.5 * unflipped[:n]
        r_tta = evaluate_seq(tta_avg, pred_df, args.gt_csv, args.pose)
        r_tta_full = evaluate_seq(postprocess_pipeline(tta_avg), pred_df, args.gt_csv, args.pose)
        rows = [
            ("baseline (no TTA)", headline(baseline)),
            ("[TTA] LR-flip average", headline(r_tta)),
            ("[TTA+postproc]", headline(r_tta_full)),
        ]
    else:
        # Self-flip 시뮬레이션 (참고용) — 같은 모델 출력을 좌우 반전 후 다시 반전
        # 실제로는 lifter가 같은 결과를 내야 동치 (identity) — 차이가 있으면 lifter가
        # 좌우 비대칭이라는 신호. 가려짐 augmentation이 부족하다는 진단도 가능.
        flipped = flip_skeleton_lr(pred_seq)
        re_flipped = flip_skeleton_lr(flipped)
        diff = float(np.linalg.norm(re_flipped - pred_seq, axis=-1).mean())
        print(f"[diag] self-flip residual = {diff:.6f} (should be ~0)")
        rows = [
            ("baseline (no TTA)", headline(baseline)),
        ]
        print()
        print("INFO: --pred-csv-flipped 없음. 실제 TTA 효과 측정을 위해서는")
        print("      좌우 반전한 입력으로 lifter를 한 번 더 돌린 별도 CSV가 필요합니다.")
        print("      라이브 모드 (onpose_v6_coach.py) 에 TTA 옵션이 추가되면 비교 가능.")

    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
