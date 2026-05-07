"""자동 품질 게이트 — raw_outputs.jsonl 을 받아 학습 가능 케이스만 추려낸다.

체크 항목:
1. 정확히 3문장 (한국어 종결어미 기준).
2. 길이 [80, 350] 자.
3. 한국어 비율 ≥ 0.6 (한글 음절이 알파벳·숫자 합보다 많아야 한다).
4. 거부 응답 단어("AI", "할 수 없", "죄송하지만") 차단.
5. Grounding: input 의 phase 이름 또는 top issue feature 이름 중 적어도 하나가 output 에 등장.
   - feature 이름은 한글 의역도 인정 (hip_angle → 고관절, knee_angle → 무릎, trunk_angle → 척추,
     hip_symmetry → 좌우 골반, knee_symmetry → 좌우 무릎, *_velocity → 속도/리듬, arm_reach → 팔)
6. 자세 이름 오기재 차단 (다른 자세의 한글명이 더 빈번히 등장).
7. 중복 차단 (output 의 단순 string 정확 일치만 — 학습 품질에 충분).

탈락한 케이스는 reports/synth_review_queue.md 로 한 줄씩 기록.

Usage:
    python filter.py --raw data/synth/raw_outputs.jsonl --out data/synth/filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

sys.path.insert(0, str(SCRIPT_DIR))
from pose_meta import POSE_BY_KEY  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "min" / "llm_finetune" / "reports"

FEATURE_KOR_HINTS = {
    "hip_angle": ["고관절", "엉덩이"],
    "knee_angle": ["무릎"],
    "trunk_angle": ["척추", "몸통", "허리"],
    "hip_symmetry": ["좌우", "대칭", "골반"],
    "knee_symmetry": ["좌우", "양쪽", "무릎"],
    "hip_velocity": ["속도", "리듬", "템포"],
    "knee_velocity": ["속도", "리듬", "템포"],
    "trunk_velocity": ["속도", "흐름"],
    "arm_reach": ["팔", "손끝"],
    "shoulder_wrist_y_delta": ["팔", "손목", "어깨"],
    "hip": ["고관절", "엉덩이"],
    "knee": ["무릎"],
}

REJECT_PATTERNS = [
    r"\bAI\b",
    r"\b인공지능\b",
    r"할\s*수\s*없",
    r"답할\s*수\s*없",
    r"죄송하지만",
    r"제공할\s*수\s*없",
    r"잘\s*모르겠",
]
REJECT_RE = re.compile("|".join(REJECT_PATTERNS))

# 종결어미 ".", "!", "?", "다.", "요." 등 기준으로 한국어 문장 분리
SENT_SPLIT_RE = re.compile(r"(?<=[다요죠까])[\.\!\?]\s+|(?<=[\.\!\?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    parts = SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip().strip(".!? ")
        if p:
            out.append(p)
    return out


def korean_ratio(text: str) -> float:
    han = sum(1 for c in text if "가" <= c <= "힣")
    other = sum(1 for c in text if c.isalnum() and not ("가" <= c <= "힣"))
    if han + other == 0:
        return 0.0
    return han / (han + other)


def grounded(item: Dict) -> Tuple[bool, str]:
    output = item["raw_output"]
    grid = item.get("grid", {})
    phase = grid.get("dom_phase")
    feature = grid.get("top_feature")
    hits = []
    if phase and phase in output:
        hits.append(f"phase:{phase}")
    if feature:
        if feature in output:
            hits.append(f"feature_en:{feature}")
        for hint in FEATURE_KOR_HINTS.get(feature, []):
            if hint in output:
                hits.append(f"feature_kr:{hint}")
                break
    return (len(hits) > 0, ",".join(hits) or "none")


def pose_name_check(item: Dict) -> Tuple[bool, str]:
    """자기 자세 이름이 다른 자세 이름보다 자주 등장하면 OK. 반대면 차단."""
    output = item["raw_output"]
    own_pose = POSE_BY_KEY[item["pose_key"]]
    own_kr = own_pose.name_kr
    own_count = output.count(own_kr)
    other_max = 0
    other_max_name = ""
    for k, p in POSE_BY_KEY.items():
        if k == item["pose_key"]:
            continue
        c = output.count(p.name_kr)
        if c > other_max:
            other_max = c
            other_max_name = p.name_kr
    if other_max > own_count and other_max > 0:
        return False, f"other-pose-confused:{other_max_name}({other_max}>{own_count})"
    return True, ""


def evaluate(item: Dict) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    text = item["raw_output"].strip()
    n_chars = len(text)

    if not (80 <= n_chars <= 350):
        reasons.append(f"length:{n_chars}")

    sents = split_sentences(text)
    if len(sents) != 3:
        reasons.append(f"sentences:{len(sents)}")

    kr = korean_ratio(text)
    if kr < 0.6:
        reasons.append(f"korean_ratio:{kr:.2f}")

    if REJECT_RE.search(text):
        reasons.append("reject_phrase")

    ok, hits = grounded(item)
    if not ok:
        reasons.append("not_grounded")

    ok, why = pose_name_check(item)
    if not ok:
        reasons.append(why)

    return (len(reasons) == 0, reasons)


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=SCRIPT_DIR / "raw_outputs.jsonl")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "filtered.jsonl")
    ap.add_argument("--review-md", type=Path, default=REPORTS_DIR / "synth_review_queue.md")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.review_md.parent.mkdir(parents=True, exist_ok=True)

    seen_outputs: set[str] = set()
    passed = 0
    rejected = 0
    reason_counter: Counter[str] = Counter()
    review_lines: List[str] = []

    with args.out.open("w", encoding="utf-8") as out_f:
        for item in iter_jsonl(args.raw):
            ok, reasons = evaluate(item)
            text = item["raw_output"].strip()
            if ok and text in seen_outputs:
                ok = False
                reasons.append("dup_exact")
            if ok:
                seen_outputs.add(text)
                record = {
                    "id": item["id"],
                    "pose_key": item["pose_key"],
                    "scorer_available": item["scorer_available"],
                    "grid": item["grid"],
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                passed += 1
            else:
                rejected += 1
                for r in reasons:
                    reason_counter[r.split(":")[0]] += 1
                review_lines.append(
                    f"- `{item['id']}` ({item['pose_key']}, scorer={item['scorer_available']}) "
                    f"reasons={reasons} text={text[:120]!r}"
                )

    md = ["# 합성 검수 큐\n"]
    md.append(f"총 raw {passed + rejected}개 중 자동 통과 {passed}개 / 탈락 {rejected}개.\n")
    md.append("## 탈락 사유 분포\n")
    for k, v in reason_counter.most_common():
        md.append(f"- {k}: {v}")
    md.append("\n## 탈락 케이스 목록\n")
    md.extend(review_lines)
    args.review_md.write_text("\n".join(md), encoding="utf-8")

    print(f"passed={passed}, rejected={rejected}")
    print(f"  → {args.out}")
    print(f"  review queue → {args.review_md}")
    print("  reasons:", dict(reason_counter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
