"""train/eval stratified split — filtered.jsonl 을 sft.jsonl + eval.jsonl 로 나눈다.

stratify 키: (pose_key, severity bin). 각 stratum 에서 eval 비율만큼 샘플링.
시드는 build_inputs.py 와 별도 — 합성 시드를 바꾸지 않고도 split 만 재현 가능.

학습 스크립트가 기대하는 3-필드 JSONL (instruction/input/output) 만 떨군다.

Usage:
    python split.py --in data/synth/filtered.jsonl --train data/sft.jsonl --eval data/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratum_key(item: Dict) -> str:
    grid = item.get("grid", {})
    return f"{item['pose_key']}|sev={grid.get('severity')}"


def to_train_record(item: Dict) -> Dict:
    return {
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=SCRIPT_DIR / "filtered.jsonl")
    ap.add_argument("--train", type=Path, default=DATA_DIR / "sft.jsonl")
    ap.add_argument("--eval", type=Path, default=DATA_DIR / "eval.jsonl")
    ap.add_argument("--eval-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_stratum: Dict[str, List[Dict]] = defaultdict(list)
    for item in iter_jsonl(args.inp):
        by_stratum[stratum_key(item)].append(item)

    train: List[Dict] = []
    eval_: List[Dict] = []
    for key, items in by_stratum.items():
        rng.shuffle(items)
        n_eval = max(1, int(round(len(items) * args.eval_fraction))) if len(items) >= 5 else 0
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)

    write_jsonl(args.train, [to_train_record(x) for x in train])
    write_jsonl(args.eval, [to_train_record(x) for x in eval_])

    print(f"strata: {len(by_stratum)}")
    print(f"train: {len(train)} → {args.train}")
    print(f"eval:  {len(eval_)} → {args.eval}")

    pose_counts = defaultdict(lambda: [0, 0])
    for x in train:
        pose_counts[x["pose_key"]][0] += 1
    for x in eval_:
        pose_counts[x["pose_key"]][1] += 1
    print("pose breakdown (train, eval):")
    for k, (t, e) in sorted(pose_counts.items()):
        print(f"  {k}: train={t}, eval={e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
