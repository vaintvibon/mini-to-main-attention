# -*- coding: utf-8 -*-
"""
Fair 4-Mini / 8-Main budget training.

핵심 수정점
-----------
1) mini_main / main_only 모두 shared budget {2,4,6,8}을 정확히 같은 빈도로 학습.
2) mini_main의 B=0은 auxiliary loss로만 추가.
   기본값 mini0_weight=0.25:
   4 batch 동안 B0 loss 총 가중치 = 1.0,
   각 shared budget의 CE 총 가중치도 = 1.0.
3) best checkpoint는 두 모델 모두 shared mean CE(B=2,4,6,8)로 선택.
   -> Mini-only B=0 때문에 target checkpoint 선택 기준이 달라지는 문제 제거.
4) 공식 CIFAR-10 test는 사용하지 않음.
5) heldout permutation [45000,50000)은 training/validation에서 사용하지 않음.

추천 본 실험:
  30 epochs / train 40000 / val 5000 / depth 4 / AMP
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


SHARED_BUDGETS = (2, 4, 6, 8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mini_main", "main_only"], required=True)
    p.add_argument("--data-dir", default="/content/cifar10")
    p.add_argument("--output-dir", default="./outputs/budgeted_v2_fair")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--main-heads", type=int, default=8)
    p.add_argument("--mini-heads", type=int, default=4)
    p.add_argument("--mini-head-dim", type=int, default=16)
    p.add_argument("--direct-k", type=int, default=2)
    p.add_argument("--pool-ratio", type=int, default=2)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # shared low-budget KD; teacher is B=8 and detached.
    # Both mini_main and main_only use exactly the same rule.
    p.add_argument("--kd-alpha", type=float, default=0.3)
    p.add_argument("--kd-temp", type=float, default=2.0)

    # Mini-only auxiliary objective.
    # Over four batches, 4 * 0.25 = 1.0, matching one exposure
    # of each shared budget in the balanced cycle.
    p.add_argument("--mini0-weight", type=float, default=0.25)

    p.add_argument("--route-tau-start", type=float, default=1.5)
    p.add_argument("--route-tau-end", type=float, default=0.5)

    p.add_argument("--amp", action="store_true")

    p.add_argument("--train-size", type=int, default=40000)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data(args):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_base = datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=train_tf
    )
    eval_base = datasets.CIFAR10(
        args.data_dir, train=True, download=False, transform=eval_tf
    )

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(50000, generator=g).tolist()

    if args.train_size > 40000:
        raise ValueError("train_size must be <= 40000.")
    if args.val_size > 5000:
        raise ValueError("val_size must be <= 5000.")

    train_idx = perm[:args.train_size]
    val_idx = perm[40000:40000 + args.val_size]

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(eval_base, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 100),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model(args):
    return BudgetedMiniMainViTV2(
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        main_heads=args.main_heads,
        mini_heads=args.mini_heads,
        mini_head_dim=args.mini_head_dim,
        direct_k=args.direct_k,
        pool_ratio=args.pool_ratio,
        mode=args.mode,
        route_tau=args.route_tau_start,
    )


def kd_loss(student, teacher, temp):
    return F.kl_div(
        F.log_softmax(student / temp, dim=-1),
        F.softmax(teacher / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)


def autocast_ctx(device, enabled):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def route_tau(epoch, epochs, start, end):
    if epochs <= 1:
        return end
    t = (epoch - 1) / (epochs - 1)
    return start * ((end / start) ** t)


def balanced_budget_for_batch(batch_idx, epoch):
    """
    Exact balanced cycle over shared budgets.
    Epoch마다 시작점을 바꿔 특정 batch position과 budget의 결합을 줄인다.
    """
    offset = (epoch - 1) % len(SHARED_BUDGETS)
    return SHARED_BUDGETS[(batch_idx + offset) % len(SHARED_BUDGETS)]


def _new_budget_stats():
    return {
        b: {"loss_sum": 0.0, "correct": 0, "n": 0, "batches": 0}
        for b in SHARED_BUDGETS
    }


def train_epoch(model, loader, optimizer, scaler, device, args, epoch):
    model.train()

    stats = _new_budget_stats()
    b0_loss_sum = 0.0
    b0_correct = 0
    b0_n = 0
    b0_batches = 0

    objective_sum = 0.0
    objective_n = 0

    for bi, (x, y) in enumerate(loader):
        if args.max_train_batches and bi >= args.max_train_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        budget = balanced_budget_for_batch(bi, epoch)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device, args.amp):
            logits = model(x, budget=budget)
            ce = F.cross_entropy(logits, y)

            loss = ce

            # Same KD rule for both models.
            # B=8 teacher is forward-only and detached, so it does not change
            # gradient exposure of the full budget.
            if args.kd_alpha > 0 and budget != 8:
                with torch.no_grad():
                    teacher_logits = model(x, budget=8)
                kd = kd_loss(logits, teacher_logits, args.kd_temp)
                loss = loss + args.kd_alpha * kd

            # Target-only Mini-only auxiliary objective.
            # Mini representation already participates in every budget.
            # This term explicitly trains B=0 without changing shared-budget frequency.
            if args.mode == "mini_main" and args.mini0_weight > 0:
                logits0 = model(x, budget=0)
                ce0 = F.cross_entropy(logits0, y)
                loss = loss + args.mini0_weight * ce0
            else:
                logits0 = None
                ce0 = None

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        n = y.numel()

        s = stats[budget]
        s["loss_sum"] += ce.item() * n
        s["correct"] += logits.argmax(-1).eq(y).sum().item()
        s["n"] += n
        s["batches"] += 1

        if logits0 is not None:
            b0_loss_sum += ce0.item() * n
            b0_correct += logits0.argmax(-1).eq(y).sum().item()
            b0_n += n
            b0_batches += 1

        objective_sum += loss.item() * n
        objective_n += n

    out = {
        "objective": objective_sum / max(objective_n, 1),
        "shared": {},
    }

    for b in SHARED_BUDGETS:
        s = stats[b]
        out["shared"][str(b)] = {
            "ce": s["loss_sum"] / max(s["n"], 1),
            "acc": 100.0 * s["correct"] / max(s["n"], 1),
            "samples": s["n"],
            "batches": s["batches"],
        }

    if args.mode == "mini_main":
        out["mini0"] = {
            "ce": b0_loss_sum / max(b0_n, 1),
            "acc": 100.0 * b0_correct / max(b0_n, 1),
            "samples": b0_n,
            "batches": b0_batches,
            "loss_weight": args.mini0_weight,
        }

    return out


@torch.no_grad()
def eval_budget(model, loader, budget, device, args):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0
    active_sum = 0
    block_samples = 0

    for bi, (x, y) in enumerate(loader):
        if args.max_val_batches and bi >= args.max_val_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx(device, args.amp):
            logits, infos = model(
                x,
                budget=budget,
                return_info=True,
            )
            loss_sum += F.cross_entropy(
                logits, y, reduction="sum"
            ).item()

        correct += logits.argmax(-1).eq(y).sum().item()
        total += y.numel()

        for info in infos:
            active_sum += int(
                info["active_main_mask"].sum().item()
            )
            block_samples += int(
                info["active_main_mask"].shape[0]
            )

    return {
        "ce": loss_sum / max(total, 1),
        "acc": 100.0 * correct / max(total, 1),
        "avg_active_main_heads": active_sum / max(block_samples, 1),
    }


@torch.no_grad()
def eval_all(model, loader, device, args):
    budgets = (
        [0, 2, 4, 6, 8]
        if args.mode == "mini_main"
        else [2, 4, 6, 8]
    )

    out = {
        str(b): eval_budget(model, loader, b, device, args)
        for b in budgets
    }

    # IMPORTANT:
    # This is the common checkpoint-selection criterion for BOTH models.
    out["shared_mean_ce"] = sum(
        out[str(b)]["ce"] for b in SHARED_BUDGETS
    ) / len(SHARED_BUDGETS)

    if args.mode == "mini_main":
        out["all_mean_ce_including_b0"] = sum(
            out[str(b)]["ce"] for b in [0, 2, 4, 6, 8]
        ) / 5.0

    return out


def save_ckpt(path, model, args, epoch, val):
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "img_size": 32,
                "patch_size": 4,
                "num_classes": 10,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "main_heads": args.main_heads,
                "mini_heads": args.mini_heads,
                "mini_head_dim": args.mini_head_dim,
                "direct_k": args.direct_k,
                "pool_ratio": args.pool_ratio,
                "mode": args.mode,
            },
            "epoch": epoch,
            "val": val,
            "args": vars(args),
            "checkpoint_selection_metric": "shared_mean_ce_B2_B4_B6_B8",
        },
        path,
    )


def main():
    args = parse_args()
    seed_all(args.seed)

    if (args.main_heads, args.mini_heads) != (8, 4):
        raise ValueError(
            "This experiment is fixed to Mini=4, Main=8."
        )
    if args.embed_dim % args.main_heads != 0:
        raise ValueError(
            "embed_dim must be divisible by main_heads."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("FAIR 4 MINI / 8 MAIN BUDGET TRAIN")
    print("=" * 100)
    print("mode:", args.mode)
    print("device:", device)
    print("shared training budgets:", SHARED_BUDGETS)
    print(
        "shared budget exposure: EXACT BALANCED CYCLE "
        "for BOTH mini_main and main_only"
    )
    if args.mode == "mini_main":
        print(
            f"B=0 auxiliary CE weight: {args.mini0_weight} "
            "(does not change shared-budget exposure)"
        )
    print(
        "best checkpoint criterion: "
        "mean validation CE over B={2,4,6,8}"
    )
    print("Official CIFAR-10 test: NOT USED")
    print("Heldout permutation [45000,50000): NOT USED")

    if device.type != "cuda":
        print(
            "WARNING: GPU is not enabled. Full 30-epoch run will be slow."
        )

    train_loader, val_loader = build_data(args)
    model = build_model(args).to(device)

    print(
        f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M"
    )

    budgets = (
        [0, 2, 4, 6, 8]
        if args.mode == "mini_main"
        else [2, 4, 6, 8]
    )
    for b in budgets:
        m = model.estimate_block_attention_macs(b)
        print(
            f"B={b}: attention MAC/block="
            f"{m['attention_total_macs']/1e6:.3f}M "
            f"(mini={m['mini_macs']/1e6:.3f}, "
            f"main={m['main_macs']/1e6:.3f})"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.amp and device.type == "cuda"),
    )

    outdir = Path(args.output_dir) / args.mode
    outdir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    best_epoch = -1
    metrics = outdir / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()

    for epoch in range(1, args.epochs + 1):
        tau = route_tau(
            epoch,
            args.epochs,
            args.route_tau_start,
            args.route_tau_end,
        )
        model.set_route_temperature(tau)

        t0 = time.time()
        tr = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
        )

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            va = eval_all(
                model,
                val_loader,
                device,
                args,
            )
        else:
            va = None

        rec = {
            "epoch": epoch,
            "tau": tau,
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
            "train": tr,
            "val": va,
        }
        with metrics.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(rec, ensure_ascii=False) + "\n"
            )

        print(
            f"\nEpoch {epoch:03d}/{args.epochs} "
            f"tau={tau:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"time={rec['sec']:.1f}s"
        )
        print(
            f"Train objective={tr['objective']:.4f}"
        )

        for b in SHARED_BUDGETS:
            s = tr["shared"][str(b)]
            print(
                f"  Train B={b}: CE={s['ce']:.4f} "
                f"Acc={s['acc']:.2f}% "
                f"batches={s['batches']}"
            )

        if args.mode == "mini_main":
            s0 = tr["mini0"]
            print(
                f"  Train B=0(aux): CE={s0['ce']:.4f} "
                f"Acc={s0['acc']:.2f}% "
                f"weight={s0['loss_weight']}"
            )

        if va is not None:
            for b in budgets:
                m = va[str(b)]
                print(
                    f"  Val B={b}: CE={m['ce']:.4f} "
                    f"Acc={m['acc']:.2f}% "
                    f"active={m['avg_active_main_heads']:.2f}"
                )

            print(
                "  shared mean CE(B2/B4/B6/B8)="
                f"{va['shared_mean_ce']:.6f}"
            )
            if args.mode == "mini_main":
                print(
                    "  all mean CE(including B0)="
                    f"{va['all_mean_ce_including_b0']:.6f}"
                )

            if va["shared_mean_ce"] < best:
                best = va["shared_mean_ce"]
                best_epoch = epoch
                save_ckpt(
                    outdir / "best.pt",
                    model,
                    args,
                    epoch,
                    va,
                )
                print("  -> BEST")

        save_ckpt(
            outdir / "last.pt",
            model,
            args,
            epoch,
            va,
        )
        scheduler.step()

    print("\n" + "=" * 100)
    print("DONE")
    print("best epoch =", best_epoch)
    print(
        "best shared mean CE(B2/B4/B6/B8) =",
        best,
    )
    print("best checkpoint:", outdir / "best.pt")
    print("Official CIFAR-10 test remains untouched.")


if __name__ == "__main__":
    main()
