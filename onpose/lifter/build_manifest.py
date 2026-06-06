from __future__ import annotations

import argparse
from pathlib import Path

from dataset import save_manifest, scan_aihub_pilates_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIHub 필라테스 2D/3D 페어 manifest 생성")
    p.add_argument("--data-root", type=Path, required=True, help="필라테스 데이터셋 루트")
    p.add_argument("--out", type=Path, required=True, help="manifest.jsonl 저장 경로")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    entries = scan_aihub_pilates_pairs(args.data_root)
    save_manifest(entries, args.out)
    print({
        "saved": str(args.out),
        "num_pairs": len(entries),
        "example": entries[0].__dict__ if entries else None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
