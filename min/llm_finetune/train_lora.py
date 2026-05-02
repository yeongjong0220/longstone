"""Gemma-2-2b-it QLoRA 파인튜닝 스크립트.

기본 사용:
    python train_lora.py \
        --data data/sft.jsonl \
        --output checkpoints/spine_seal_v1 \
        --epochs 3

dry-run (더미 데이터로 파이프라인 동작 확인):
    python train_lora.py --data data/dummy.jsonl --output checkpoints/dryrun --epochs 1

설정값은 RTX 3050 8GB에서 OOM 없이 도는 것을 기준으로 한다. 더 큰 GPU에서는
batch_size / max_seq_length / r 을 늘릴 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} JSON 파싱 실패: {e}") from e
    if not rows:
        raise ValueError(f"{path}에 학습 샘플이 없습니다.")
    for i, r in enumerate(rows):
        for k in ("instruction", "input", "output"):
            if k not in r:
                raise ValueError(f"{path}[{i}] 필드 '{k}' 누락")
    return rows


def to_chat(rows: list[dict], tokenizer) -> Dataset:
    formatted = []
    for r in rows:
        user_text = f"{r['instruction']}\n\n{r['input']}"
        chat = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": r["output"]},
        ]
        formatted.append({"text": tokenizer.apply_chat_template(chat, tokenize=False)})
    return Dataset.from_list(formatted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data)
    print(f"학습 샘플 수: {len(rows)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"베이스 모델 로딩: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = to_chat(rows, tokenizer)

    cfg = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=cfg,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\n✅ 어댑터 저장 완료: {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
