
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import PilatesTemporalLifterDataset, load_manifest, split_by_actor, PHASE_NAMES_3, PHASE_NAMES_4
from losses import compute_losses
from model import TemporalLifterConfig, TemporalLifterWithPhaseHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="필라테스 2D→3D temporal lifter + phase head 학습")
    p.add_argument("--manifest", type=Path, required=True, help="build_manifest.py 결과 jsonl")
    p.add_argument("--outdir", type=Path, required=True, help="체크포인트/로그 출력 디렉토리")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-size", type=int, default=81)
    p.add_argument("--stride", type=int, default=27)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--phase-scheme", choices=["progress3", "cyclic4"], default="progress3")
    p.add_argument("--non-causal", action="store_true", help="future frame까지 보는 symmetric conv 사용")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--w-pose", type=float, default=1.0)
    p.add_argument("--w-phase", type=float, default=0.2)
    p.add_argument("--w-bone", type=float, default=0.2)
    p.add_argument("--w-vel", type=float, default=0.1)
    p.add_argument("--w-angle", type=float, default=0.1)
    # 사용자 요청: train 비율 최대 + val 만 몇 명만 남기기
    p.add_argument("--val-ratio", type=float, default=0.10, help="actor split — val 비율 (default 0.10)")
    p.add_argument("--test-ratio", type=float, default=0.05, help="actor split — test 비율 (default 0.05)")
    p.add_argument("--pose-filter", type=str, default=None,
                   help="쉼표 구분 pose 이름만 사용 (e.g. 'Bridging,Spine Stretch,The Seal')")
    # 사용자 요청: 동작당 actor 1명씩만 test 로 빼는 per-pose stratified split
    p.add_argument("--per-pose-split", action="store_true",
                   help="동작별로 actor N명을 test/val 로 빼는 stratified split (per_pose 모드)")
    p.add_argument("--test-per-pose", type=int, default=1,
                   help="per-pose 모드에서 동작당 test actor 수 (default 1)")
    p.add_argument("--val-per-pose", type=int, default=1,
                   help="per-pose 모드에서 동작당 val actor 수 (default 1)")
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y3d = torch.stack([b["y3d"] for b in batch], dim=0)
    phase = torch.stack([b["phase"] for b in batch], dim=0)
    category = torch.stack([b["category"] for b in batch], dim=0)
    level = torch.stack([b["level"] for b in batch], dim=0)
    meta = [b["meta"] for b in batch]
    return {"x": x, "y3d": y3d, "phase": phase, "category": category, "level": level, "meta": meta}


@torch.no_grad()
def evaluate(
    model: TemporalLifterWithPhaseHead,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "pose": 0.0, "phase": 0.0, "bone": 0.0, "vel": 0.0, "angle": 0.0, "mpjpe": 0.0}
    count = 0
    phase_correct = 0
    phase_total = 0

    skipped = 0
    for batch in loader:
        x = batch["x"].to(device)
        y3d = batch["y3d"].to(device)
        phase = batch["phase"].to(device)

        if not torch.isfinite(x).all() or not torch.isfinite(y3d).all():
            skipped += 1
            continue

        pred_3d, phase_logits = model(x)
        losses = compute_losses(pred_3d, y3d, phase_logits, phase, **loss_weights)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            skipped += 1
            continue

        bs = x.shape[0]
        count += bs
        for k in sums:
            sums[k] += float(losses[k].item()) * bs

        pred_phase = phase_logits.argmax(dim=-1)
        phase_correct += int((pred_phase == phase).sum().item())
        phase_total += int(phase.numel())

    metrics = {k: v / max(count, 1) for k, v in sums.items()}
    metrics["phase_acc"] = phase_correct / max(phase_total, 1)
    metrics["skipped_batches"] = float(skipped)
    return metrics


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(args.manifest)

    # 사용자 요청: 특정 동작들만 필터링 (3 동작 모두 사용)
    if args.pose_filter:
        keep = {p.strip() for p in args.pose_filter.split(",") if p.strip()}
        n_before = len(entries)
        entries = [e for e in entries if e.category_2 in keep]
        print(f"[filter] poses {sorted(keep)} : {n_before} → {len(entries)} entries")

    if args.per_pose_split:
        # 동작별 stratified split — 각 동작에서 actor N명을 test, N명을 val 로
        import collections, random
        by_pose: dict = collections.defaultdict(list)
        for e in entries:
            by_pose[e.category_2].append(e)
        rng = random.Random(args.seed)
        train_entries, val_entries, test_entries = [], [], []
        split_log = {}
        for pose, pose_entries in by_pose.items():
            actors = sorted({e.actor for e in pose_entries})
            rng.shuffle(actors)
            n_t = min(args.test_per_pose, max(0, len(actors) - 2))
            n_v = min(args.val_per_pose, max(0, len(actors) - n_t - 1))
            test_actors = set(actors[:n_t])
            val_actors = set(actors[n_t:n_t + n_v])
            train_actors = set(actors[n_t + n_v:])
            for e in pose_entries:
                if e.actor in test_actors:
                    test_entries.append(e)
                elif e.actor in val_actors:
                    val_entries.append(e)
                else:
                    train_entries.append(e)
            split_log[pose] = {
                "train_actors": sorted(train_actors),
                "val_actors": sorted(val_actors),
                "test_actors": sorted(test_actors),
            }
        for pose, d in split_log.items():
            print(f"[per-pose split] {pose:20s}  "
                  f"train={len(d['train_actors'])}  "
                  f"val={d['val_actors']}  "
                  f"test={d['test_actors']}")
        # outdir 에 split 기록 저장 (재현성)
        with (args.outdir / "split_actors.json").open("w", encoding="utf-8") as f:
            json.dump(split_log, f, indent=2, ensure_ascii=False)
    else:
        train_entries, val_entries, test_entries = split_by_actor(
            entries,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    print(f"[split] actors → train={len({e.actor for e in train_entries})}, "
          f"val={len({e.actor for e in val_entries})}, "
          f"test={len({e.actor for e in test_entries})}")
    print(f"[split] clips  → train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")

    train_ds = PilatesTemporalLifterDataset(
        train_entries,
        window_size=args.window_size,
        stride=args.stride,
        phase_scheme=args.phase_scheme,
        augment_occlusion=True,
        seed=args.seed,
    )
    val_ds = PilatesTemporalLifterDataset(
        val_entries,
        window_size=args.window_size,
        stride=args.stride,
        phase_scheme=args.phase_scheme,
        augment_occlusion=False,
        seed=args.seed,
    )
    test_ds = PilatesTemporalLifterDataset(
        test_entries,
        window_size=args.window_size,
        stride=args.stride,
        phase_scheme=args.phase_scheme,
        augment_occlusion=False,
        seed=args.seed,
    )

    num_phases = 3 if args.phase_scheme == "progress3" else 4
    phase_names = PHASE_NAMES_3 if num_phases == 3 else PHASE_NAMES_4

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    cfg = TemporalLifterConfig(
        num_joints=15,
        in_features=3,
        hidden_dim=args.hidden_dim,
        causal=not args.non_causal,
        num_phases=num_phases,
    )
    model = TemporalLifterWithPhaseHead(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    device = torch.device(args.device)
    loss_weights = {
        "w_pose": args.w_pose,
        "w_phase": args.w_phase,
        "w_bone": args.w_bone,
        "w_vel": args.w_vel,
        "w_angle": args.w_angle,
    }

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums = {"total": 0.0, "pose": 0.0, "phase": 0.0, "bone": 0.0, "vel": 0.0, "angle": 0.0, "mpjpe": 0.0}
        count = 0
        skipped_train_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y3d = batch["y3d"].to(device)
            phase = batch["phase"].to(device)

            if not torch.isfinite(x).all() or not torch.isfinite(y3d).all():
                skipped_train_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            pred_3d, phase_logits = model(x)
            losses = compute_losses(pred_3d, y3d, phase_logits, phase, **loss_weights)
            if not all(torch.isfinite(v).all() for v in losses.values()):
                skipped_train_batches += 1
                continue

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                skipped_train_batches += 1
                continue

            optimizer.step()

            bs = x.shape[0]
            count += bs
            for k in train_sums:
                train_sums[k] += float(losses[k].item()) * bs

        scheduler.step()
        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in train_sums.items()}
        train_metrics['train_skipped_batches'] = float(skipped_train_batches)
        val_metrics = evaluate(model, val_loader, device, loss_weights) if len(val_ds) > 0 else {}
        test_metrics = evaluate(model, test_loader, device, loss_weights) if len(test_ds) > 0 else {}

        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update(train_metrics)
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        history.append(row)

        val_score = val_metrics.get("total", train_metrics["train_total"])
        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "phase_scheme": args.phase_scheme,
                    "phase_names": phase_names,
                    "class_to_idx": train_ds.class_to_idx,
                    "level_to_idx": train_ds.level_to_idx,
                    "loss_weights": loss_weights,
                },
                args.outdir / "best.pt",
            )

        print(json.dumps(row, ensure_ascii=False))

    with (args.outdir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with (args.outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_train_windows": len(train_ds),
                "num_val_windows": len(val_ds),
                "num_test_windows": len(test_ds),
                "num_classes": len(train_ds.class_to_idx),
                "phase_scheme": args.phase_scheme,
                "phase_names": phase_names,
                "loss_weights": loss_weights,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
