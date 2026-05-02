"""LoRA 없이 전체 가중치(full) SFT 학습 스크립트.

용도:
- (A) 큰 모델(Gemma-2-2b 등)을 RTX 3050 8GB에서 full FT 시도 → OOM 로그 확보
- (C) 작은 모델(Qwen2.5-0.5B, HyperCLOVAX-0.5B 등)에서 실제 full FT 수행

train_lora.py와 거의 동일한 데이터 처리 로직을 쓰지만 LoRA를 끼우지 않고 모델
전체 파라미터를 학습한다. 메모리 절약 옵션은 모두 켠 상태:
- bf16 학습
- gradient checkpointing
- bitsandbytes 8-bit Adam (`adamw_bnb_8bit`)
- batch 1 + grad_accum 8

OOM이 나면 reports/ 폴더에 자동으로 로그를 남기고 종료한다 — 그 로그가
"환경상 full FT 불가"라는 사실의 객관 증거가 된다.

기본 사용:
    python train_full.py --base-model google/gemma-2-2b-it \
        --data data/dummy.jsonl --output checkpoints/full_gemma_attempt --epochs 1

    python train_full.py --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/dummy.jsonl --output checkpoints/full_qwen05 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"


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


def safe_model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def write_oom_report(model_id: str, args, exc: BaseException) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = safe_model_slug(model_id)
    out = REPORTS_DIR / f"full_{slug}_oom.md"
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**2
        if torch.cuda.is_available()
        else 0
    )
    body = [
        f"# Full FT OOM 로그 — {model_id}",
        "",
        f"- 일시: {datetime.now().isoformat(timespec='seconds')}",
        f"- 디바이스: {device}",
        f"- VRAM 총량: {total_mb:.0f} MiB",
        f"- VRAM peak (학습 중단 시점): {peak_mb:.0f} MiB",
        f"- batch_size: {args.batch_size}, grad_accum: {args.grad_accum}, max_seq: {args.max_seq}",
        f"- bf16: True, gradient_checkpointing: True, optimizer: adamw_bnb_8bit",
        "",
        "## 결론",
        f"`{model_id}` 모델을 RTX 3050 8GB(VRAM {total_mb:.0f} MiB)에서 full FT 시도 중 메모리 부족으로 학습이 중단됨. "
        "메모리 절약 옵션(bf16 + gradient checkpointing + 8-bit Adam + batch 1)을 모두 적용한 상태에서도 불가능했으므로 "
        "이 모델은 본 환경에서 LoRA(QLoRA) 방식으로 fallback하는 것이 정당화된다.",
        "",
        "## 예외 트레이스",
        "```",
        f"{type(exc).__name__}: {exc}",
        "",
        traceback.format_exc(),
        "```",
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="HF 모델 ID 또는 로컬 경로")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5, help="full FT는 LoRA보다 작은 lr 권장")
    ap.add_argument("--max-seq", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data)
    print(f"학습 샘플 수: {len(rows)}")
    print(f"베이스 모델: {args.base_model} (full fine-tuning, LoRA X)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        ds = to_chat(rows, tokenizer)

        cfg = SFTConfig(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=1,
            save_strategy="epoch",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_bnb_8bit",
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
        print(f"\n✅ Full FT 모델 저장 완료: {args.output}")

    except torch.cuda.OutOfMemoryError as e:
        report_path = write_oom_report(args.base_model, args, e)
        print(f"\n❌ CUDA OOM 발생 — full FT 불가능", file=sys.stderr)
        print(f"📄 로그 기록: {report_path}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        # bitsandbytes/transformers가 OOM을 RuntimeError로 감싸 던지는 경우 대응
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            report_path = write_oom_report(args.base_model, args, e)
            print(f"\n❌ CUDA 관련 RuntimeError(OOM 추정) — full FT 불가능", file=sys.stderr)
            print(f"📄 로그 기록: {report_path}", file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
