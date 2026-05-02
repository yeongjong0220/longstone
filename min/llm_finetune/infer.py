"""단일 추론 헬퍼. base 또는 base+adapter 출력 비교용.

사용:
    # base 모델만
    python infer.py --input-file data/dummy.jsonl --n 3

    # 어댑터 적용
    python infer.py --input-file data/dummy.jsonl --adapter checkpoints/dryrun --n 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(tokenizer, instruction: str, input_text: str) -> str:
    user_text = f"{instruction}\n\n{input_text}"
    chat = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True, help="JSONL with instruction/input fields")
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--n", type=int, default=3, help="앞에서 몇 개 샘플을 추론할지")
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: args.n]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    if args.adapter is not None:
        print(f"어댑터 로드: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    for i, r in enumerate(rows, 1):
        prompt = build_prompt(tokenizer, r["instruction"], r["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt_ms = (time.time() - t0) * 1000
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        print("=" * 70)
        print(f"[{i}/{len(rows)}] {dt_ms:.0f}ms · peak {peak_mb:.0f}MiB")
        print("입력:", r["input"].replace("\n", " | "))
        if "output" in r:
            print("정답:", r["output"])
        print("출력:", text)


if __name__ == "__main__":
    main()
