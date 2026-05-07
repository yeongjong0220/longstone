"""Teacher LLM 호출 — Gemini 2.5 Flash 로 코칭 output 생성.

`build_inputs.py` 가 만든 inputs.jsonl 을 읽어 한 줄당 한 번 Gemini 를 호출하고,
결과를 raw_outputs.jsonl 로 떨군다. 실패 케이스는 raw_outputs_errors.jsonl 에 따로 기록해
재실행 시 누락된 id 만 재시도할 수 있게 한다.

API 키는 `live_ai_coach_v4_bundle/.env` 의 `GOOGLE_API_KEY` 또는 환경변수에서.

Usage:
    python generate_outputs.py --inputs data/synth/inputs.jsonl --out data/synth/raw_outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

sys.path.insert(0, str(SCRIPT_DIR))
from pose_meta import SHARED_INSTRUCTION  # noqa: E402

GEMINI_MODEL = "gemini-2.5-flash"
ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"

DUMMY_PATH = PROJECT_ROOT / "min" / "llm_finetune" / "data" / "dummy.jsonl"


def load_few_shot() -> str:
    """기존 dummy.jsonl 6쌍을 톤 가이드로 system 에 삽입."""
    lines = []
    if not DUMMY_PATH.exists():
        return ""
    with DUMMY_PATH.open(encoding="utf-8") as f:
        for raw in f:
            r = json.loads(raw)
            lines.append(f"INPUT:\n{r['input']}\nOUTPUT:\n{r['output']}\n")
    return "\n---\n".join(lines)


SYSTEM_PROMPT_TMPL = """You are a precise but encouraging Pilates coach. Always answer in Korean (존댓말).

Hard rules — violating any one is a failure:
1. Output exactly 3 sentences.
2. Sentence 1: 전반적 품질 한 줄 평가.
3. Sentence 2: top issue feature 또는 phase 기반 가장 중요한 교정점 (자세 특성에 맞는 구체적 동작 지시 포함).
4. Sentence 3: 다음 시도에 적용할 단순 큐.
5. 분석 요약에 등장한 phase 이름이나 top issue feature 이름 중 적어도 하나를 자연스럽게 인용 (영문 그대로 또는 한글 의역 모두 가능).
6. 구현 디테일(Mahalanobis, scoring, threshold 등) 언급 금지.
7. 의학적 추론이나 통증 진단 금지.
8. 단순 출력만 — preamble("좋습니다. 분석..."), 번호, 마크다운 헤더, 영문 단락 모두 금지.

Tone reference (do not copy verbatim, only emulate):
{few_shot}
"""


def build_user_prompt(item: Dict) -> str:
    cfg_note = (
        "Scoring mode: phase-wise Mahalanobis on lifted 3D kinematic features."
        if item.get("scorer_available")
        else "Scoring mode: no trained scorer for this pose; fallback angle deviation notes only."
    )
    return (
        f"Pose: {item['pose_name_en']} ({item['pose_name_kr']})\n"
        f"{cfg_note}\n"
        f"Analysis:\n{item['input']}\n\n"
        "Write the 3-sentence Korean feedback now."
    )


def call_gemini(api_key: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.7) -> str:
    api_url = ENDPOINT_TMPL.format(m=GEMINI_MODEL, k=api_key)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxOutputTokens": 400,
        },
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt - 1))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5)
    raise RuntimeError(f"Gemini failed after retries: {last_exc}")


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {item["id"] for item in iter_jsonl(path)}


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if key and key != "PASTE_YOUR_KEY_HERE":
        return key
    env_path = PROJECT_ROOT / "live_ai_coach_v4_bundle" / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw.startswith("GOOGLE_API_KEY="):
                v = raw.split("=", 1)[1].strip().strip('"').strip("'")
                if v and v != "PASTE_YOUR_KEY_HERE":
                    return v
    raise RuntimeError(
        "GOOGLE_API_KEY 가 환경변수에도, live_ai_coach_v4_bundle/.env 에도 없습니다."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=SCRIPT_DIR / "inputs.jsonl")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "raw_outputs.jsonl")
    ap.add_argument("--errors-out", type=Path, default=SCRIPT_DIR / "raw_outputs_errors.jsonl")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sleep", type=float, default=0.4, help="요청 간 대기 (초). 무료 티어 분당 한도 회피용.")
    ap.add_argument("--limit", type=int, default=0, help="dry-run 시 N개만 호출")
    args = ap.parse_args()

    api_key = get_api_key()
    few_shot = load_few_shot()
    system_prompt = SYSTEM_PROMPT_TMPL.format(few_shot=few_shot or "(no examples)")

    done_ids = load_existing_ids(args.out)
    print(f"Already-done outputs: {len(done_ids)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.errors_out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("a", encoding="utf-8")
    err_f = args.errors_out.open("a", encoding="utf-8")

    total = 0
    written = 0
    errors = 0
    try:
        for item in iter_jsonl(args.inputs):
            total += 1
            if args.limit and (written + errors) >= args.limit:
                break
            if item["id"] in done_ids:
                continue
            user_prompt = build_user_prompt(item)
            try:
                text = call_gemini(api_key, system_prompt, user_prompt, temperature=args.temperature)
            except Exception as exc:
                errors += 1
                err_record = {"id": item["id"], "error": str(exc)}
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"[ERR] {item['id']}: {exc}")
                continue

            record = {
                "id": item["id"],
                "pose_key": item["pose_key"],
                "scorer_available": item["scorer_available"],
                "grid": item["grid"],
                "instruction": item["instruction"],
                "input": item["input"],
                "raw_output": text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if written % 25 == 0:
                print(f"  wrote {written} (errors {errors})")
    finally:
        out_f.close()
        err_f.close()

    print(f"Done. inputs scanned={total}, new written={written}, errors={errors}")
    print(f"  outputs → {args.out}")
    print(f"  errors  → {args.errors_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
