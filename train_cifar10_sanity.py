# train_cifar10_sanity.py

import argparse
import random
import time
from typing import Dict, List

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
):
    """
    CIFAR-10을 ViT 입력 크기인 224x224로 resize한다.

    주의:
        이건 sanity check용이다.
        CIFAR-10 원본은 32x32라서 224 resize가 최적 세팅은 아니다.
        목적은 구조 검증과 학습 루프 확인이다.
    """

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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
            transforms.Resize((224, 224)),
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
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    diversity_loss_fn: nn.Module,
    device: torch.device,
    budget_list: List[int],
    lambda_div: float,
    max_train_batches: int,
    epoch: int,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_div = 0.0
    total_acc = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        budget = random.choice(budget_list)

        optimizer.zero_grad(set_to_none=True)

        logits, info_list = model(
            images,
            budget=budget,
            return_info=True,
        )

        ce_loss = ce_loss_fn(logits, targets)
        div_loss = compute_diversity_loss(info_list, diversity_loss_fn)

        loss = ce_loss + lambda_div * div_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        acc = accuracy_top1(logits, targets)

        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_div += div_loss.item() * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx:04d} | "
                f"budget={budget} | "
                f"loss={loss.item():.4f} | "
                f"ce={ce_loss.item():.4f} | "
                f"div={div_loss.item():.4f} | "
                f"acc={acc:.2f}%"
            )

    elapsed = time.time() - start_time

    return {
        "loss": total_loss / max(total_samples, 1),
        "ce": total_ce / max(total_samples, 1),
        "div": total_div / max(total_samples, 1),
        "acc": total_acc / max(total_samples, 1),
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    budget_list: List[int],
    max_eval_batches: int,
):
    model.eval()

    results = {}

    for budget in budget_list:
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            logits = model(
                images,
                budget=budget,
                return_info=False,
            )

            batch_size = images.size(0)
            acc = accuracy_top1(logits, targets)

            total_acc += acc * batch_size
            total_samples += batch_size

        results[budget] = total_acc / max(total_samples, 1)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-subset", type=int, default=512)
    parser.add_argument("--test-subset", type=int, default=256)

    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lambda-div", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)

    # local sanity에서는 depth=2 권장.
    # Colab에서 조금 더 길게 볼 때 depth=12로 바꿀 수 있다.
    parser.add_argument("--depth", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    budget_list = [0, 1, 2, 3]

    train_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    model = MiniGuidedViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=args.depth,
        main_heads=3,
        mlp_ratio=4.0,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=128,
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

    print("\nModel config:")
    print(f"depth: {args.depth}")
    print("embed_dim: 192")
    print("main_heads: 3")
    print("budget_list:", budget_list)
    print(f"lambda_div: {args.lambda_div}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            diversity_loss_fn=diversity_loss_fn,
            device=device,
            budget_list=budget_list,
            lambda_div=args.lambda_div,
            max_train_batches=args.max_train_batches,
            epoch=epoch,
        )

        eval_results = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            budget_list=budget_list,
            max_eval_batches=args.max_eval_batches,
        )

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} summary")
        print(
            f"train_loss={train_stats['loss']:.4f} | "
            f"ce={train_stats['ce']:.4f} | "
            f"div={train_stats['div']:.4f} | "
            f"train_acc={train_stats['acc']:.2f}% | "
            f"time={train_stats['time']:.1f}s"
        )

        eval_str = " | ".join(
            [f"B={b}: {acc:.2f}%" for b, acc in eval_results.items()]
        )
        print("eval:", eval_str)
        print("=" * 80)

    print("\nCIFAR-10 sanity training finished.")


if __name__ == "__main__":
    main()