"""파인튜닝 시작 전 환경 점검 스크립트.

확인 항목:
- CUDA / GPU / VRAM
- HF cache 위치와 잔여 디스크
- google/gemma-2-2b-it 4-bit 양자화 로딩 smoke test (모델 다운로드 포함)

실행: python env_check.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


MODEL_ID = "google/gemma-2-2b-it"


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_torch_cuda() -> None:
    section("1. PyTorch / CUDA")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"device: {torch.cuda.get_device_name(idx)}")
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"total VRAM: {total:.2f} GiB")
    else:
        print("⚠️  CUDA 사용 불가. GPU 학습이 불가능합니다.")


def check_hf_cache() -> None:
    section("2. Hugging Face cache & 디스크")
    cache = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
    print(f"HF_HOME: {cache}")
    drive = Path(cache).anchor or "C:\\"
    usage = shutil.disk_usage(drive)
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    print(f"드라이브 {drive} 잔여 공간: {free_gb:.1f} GiB / {total_gb:.1f} GiB")
    if free_gb < 15:
        print("⚠️  잔여 공간 부족 위험. HF_HOME 환경변수로 큰 드라이브로 이동 권장.")


def check_bnb_load() -> None:
    section("3. Gemma 4-bit 로딩 smoke test")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"❌ transformers/bitsandbytes 미설치: {e}")
        print("→ pip install -r requirements.txt")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"모델: {MODEL_ID} (최초 실행 시 5GB 다운로드)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"✅ 4-bit 로딩 성공. peak VRAM: {peak_mb:.0f} MiB")
        print(f"   eos_token_id: {tokenizer.eos_token_id}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ 로딩 실패: {type(e).__name__}: {e}")
        print("→ HF 토큰 로그인 필요할 수 있음: huggingface-cli login")


def main() -> None:
    check_torch_cuda()
    check_hf_cache()
    check_bnb_load()
    print("\n점검 완료.")


if __name__ == "__main__":
    main()
