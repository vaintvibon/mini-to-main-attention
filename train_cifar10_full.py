# train_cifar10_full.py

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.mini_guided_vit import MiniGuidedViT
from losses.diversity_loss import HeadDiversityLoss


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total * 100.0


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train_subset: int,
    test_subset: int,
    img_size: int,
):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_tf,
    )

    if train_subset > 0:
        train_set = Subset(train_set, list(range(min(train_subset, len(train_set)))))

    if test_subset > 0:
        test_set = Subset(test_set, list(range(min(test_subset, len(test_set)))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


def compute_diversity_loss(
    info_list: List[Dict],
    diversity_loss_fn: nn.Module,
) -> torch.Tensor:
    div_terms = []

    for info in info_list:
        div = diversity_loss_fn(
            head_out=info["head_out"],
            active_mask=info["active_mask"],
            direct_mask=info["direct_mask"],
            mixed_mask=info["mixed_mask"],
        )
        div_terms.append(div)

    return torch.stack(div_terms).mean()


def save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv(row: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_acc: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))

    return start_epoch, best_acc


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    diversity_loss_fn: nn.Module,
    device: torch.device,
    budget_list: List[int],
    lambda_div: float,
    epoch: int,
    max_train_batches: int,
    amp: bool,
    scaler,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_div = 0.0
    total_acc = 0.0
    total_samples = 0

    start_time = time.time()

    use_diversity = lambda_div > 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        budget = random.choice(budget_list)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            if use_diversity:
                logits, info_list = model(
                    images,
                    budget=budget,
                    return_info=True,
                )
            else:
                logits = model(
                    images,
                    budget=budget,
                    return_info=False,
                )
                info_list = None

            ce_loss = ce_loss_fn(logits, targets)

            if use_diversity:
                div_loss = compute_diversity_loss(info_list, diversity_loss_fn)
            else:
                div_loss = logits.new_tensor(0.0)

            loss = ce_loss + lambda_div * div_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        acc = accuracy_top1(logits.detach(), targets)

        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_div += div_loss.item() * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                f"budget={budget} | "
                f"loss={loss.item():.4f} | "
                f"ce={ce_loss.item():.4f} | "
                f"div={div_loss.item():.4f} | "
                f"acc={acc:.2f}%"
            )

    elapsed = time.time() - start_time

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "train_ce": total_ce / max(total_samples, 1),
        "train_div": total_div / max(total_samples, 1),
        "train_acc": total_acc / max(total_samples, 1),
        "train_time": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    ce_loss_fn: nn.Module,
    device: torch.device,
    budget_list: List[int],
    max_eval_batches: int,
    amp: bool,
):
    model.eval()

    results = {}

    for budget in budget_list:
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(
                    images,
                    budget=budget,
                    return_info=False,
                )
                loss = ce_loss_fn(logits, targets)

            batch_size = images.size(0)
            acc = accuracy_top1(logits, targets)

            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size

        results[budget] = {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_acc / max(total_samples, 1),
        }

    return results


def parse_budget_list(budget_str: str) -> List[int]:
    budgets = [int(x.strip()) for x in budget_str.split(",") if x.strip() != ""]

    if len(budgets) == 0:
        raise ValueError("budget_list cannot be empty.")

    return budgets


def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/cifar10_full")
    parser.add_argument("--resume", type=str, default="")

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    # subset/debug controls
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)

    # model
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--main-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--mini-heads", type=int, default=1)
    parser.add_argument("--mini-dim", type=int, default=64)
    parser.add_argument("--pool-ratio", type=int, default=2)
    parser.add_argument("--direct-ratio", type=float, default=0.34)
    parser.add_argument("--alpha-direct", type=float, default=1.0)
    parser.add_argument("--alpha-mixed", type=float, default=0.2)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--allocator-hidden-dim", type=int, default=128)

    # budget/loss
    parser.add_argument("--budget-list", type=str, default="0,1,2,3")
    parser.add_argument("--lambda-div", type=float, default=0.0)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), output_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    print("Device:", device)
    print("AMP:", amp)

    budget_list = parse_budget_list(args.budget_list)

    print("\nExperiment config:")
    print(f"output_dir: {output_dir}")
    print(f"data_dir: {args.data_dir}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"img_size: {args.img_size}")
    print(f"patch_size: {args.patch_size}")
    print(f"depth: {args.depth}")
    print(f"embed_dim: {args.embed_dim}")
    print(f"main_heads: {args.main_heads}")
    print(f"budget_list: {budget_list}")
    print(f"lambda_div: {args.lambda_div}")

    train_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        img_size=args.img_size,
    )

    model = MiniGuidedViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        main_heads=args.main_heads,
        mlp_ratio=args.mlp_ratio,
        mini_heads=args.mini_heads,
        mini_dim=args.mini_dim,
        pool_ratio=args.pool_ratio,
        direct_ratio=args.direct_ratio,
        alpha_direct=args.alpha_direct,
        alpha_mixed=args.alpha_mixed,
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        allocator_hidden_dim=args.allocator_hidden_dim,
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    diversity_loss_fn = HeadDiversityLoss(
        mode="direct_mixed",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"start_epoch: {start_epoch}")
        print(f"best_acc: {best_acc:.2f}%")

    log_path = output_dir / "log.csv"
    last_ckpt_path = output_dir / "checkpoint_last.pth"
    best_ckpt_path = output_dir / "checkpoint_best.pth"

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            diversity_loss_fn=diversity_loss_fn,
            device=device,
            budget_list=budget_list,
            lambda_div=args.lambda_div,
            epoch=epoch,
            max_train_batches=args.max_train_batches,
            amp=amp,
            scaler=scaler,
        )

        eval_results = evaluate(
            model=model,
            test_loader=test_loader,
            ce_loss_fn=ce_loss_fn,
            device=device,
            budget_list=budget_list,
            max_eval_batches=args.max_eval_batches,
            amp=amp,
        )

        scheduler.step()

        # full-budget accuracy를 best 기준으로 사용
        full_budget = max(budget_list)
        full_budget_acc = eval_results[full_budget]["acc"]

        is_best = full_budget_acc > best_acc

        if is_best:
            best_acc = full_budget_acc

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["train_loss"],
            "train_ce": train_stats["train_ce"],
            "train_div": train_stats["train_div"],
            "train_acc": train_stats["train_acc"],
            "train_time": train_stats["train_time"],
            "best_acc": best_acc,
        }

        for b in budget_list:
            row[f"eval_b{b}_loss"] = eval_results[b]["loss"]
            row[f"eval_b{b}_acc"] = eval_results[b]["acc"]

        append_csv(row, log_path)

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_acc=best_acc,
            args=args,
        )

        if is_best:
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_acc=best_acc,
                args=args,
            )

        print("\n" + "=" * 100)
        print(f"Epoch {epoch} summary")
        print(
            f"train_loss={train_stats['train_loss']:.4f} | "
            f"ce={train_stats['train_ce']:.4f} | "
            f"div={train_stats['train_div']:.4f} | "
            f"train_acc={train_stats['train_acc']:.2f}% | "
            f"time={train_stats['train_time']:.1f}s | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        eval_str = " | ".join(
            [
                f"B={b}: loss={eval_results[b]['loss']:.4f}, acc={eval_results[b]['acc']:.2f}%"
                for b in budget_list
            ]
        )

        print("eval:", eval_str)
        print(f"best full-budget acc: {best_acc:.2f}%")
        print(f"saved last checkpoint: {last_ckpt_path}")

        if is_best:
            print(f"saved best checkpoint: {best_ckpt_path}")

        print("=" * 100)

    print("\nCIFAR-10 full training finished.")


if __name__ == "__main__":
    main()