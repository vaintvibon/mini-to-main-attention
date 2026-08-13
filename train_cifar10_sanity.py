# train_cifar10_sanity.py
"""
Strong CIFAR-10 sanity experiment for Mini-to-Main Attention.

이 단계의 목적
--------------
최종 논문 성능 실험이 아니라, 현재 prototype이 더 충분한 데이터/학습 시간에서도
일관된 경향을 보이는지 확인한다.

기본 설정:
    - CIFAR-10 train 전체 50,000
    - CIFAR-10 test 전체 10,000
    - depth=2 유지
    - embed_dim=192
    - main_heads=3
    - budgets=[0,1,2,3]
    - balanced budget exposure
    - Gumbel-ST learned allocation
    - 10 epochs

검증할 핵심:
    1. task loss / accuracy가 안정적으로 개선되는가
    2. allocator gradient가 계속 살아 있는가
    3. allocator collapse가 발생하지 않는가
    4. B=1 learned가 fixed/random보다 지속적으로 좋은가
    5. B=2에서도 learned advantage가 생기는가
    6. B=0 < B=1 <= B=2 <= B=3 경향이 형성되는가
    7. main_out norm이 budget에 따라 안정적으로 증가하는가

주의:
    fixed/random은 여기서도 inference-time ablation이다.
    논문 최종 baseline에서는 fixed/random 방식으로 별도 학습한 모델 비교가 필요하다.
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from losses.diversity_loss import HeadDiversityLoss
from models.mini_guided_vit import MiniGuidedViT


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device_arg)


def global_grad_norm_from_named_parameters(
    named_parameters: Iterable,
    name_contains: Optional[str] = None,
) -> float:
    sq_sum = 0.0
    found = False

    for name, p in named_parameters:
        if name_contains is not None and name_contains not in name:
            continue

        if p.grad is None:
            continue

        grad = p.grad.detach()

        if not torch.isfinite(grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected: {name}"
            )

        sq_sum += grad.float().pow(2).sum().item()
        found = True

    return math.sqrt(sq_sum) if found else 0.0


def build_balanced_budget_schedule(
    num_batches: int,
    budgets: List[int],
) -> List[int]:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    repeats = math.ceil(num_batches / len(budgets))
    schedule = (budgets * repeats)[:num_batches]

    random.shuffle(schedule)

    return schedule


def gumbel_temperature(
    epoch_idx: int,
    epochs: int,
    tau_start: float,
    tau_end: float,
) -> float:
    if epochs <= 1:
        return tau_end

    ratio = epoch_idx / float(epochs - 1)

    return float(
        tau_start * ((tau_end / tau_start) ** ratio)
    )


def parse_fixed_head_order(
    text: str,
    main_heads: int,
) -> List[int]:
    if text.strip() == "":
        order = list(range(main_heads))
    else:
        order = [
            int(x.strip())
            for x in text.split(",")
            if x.strip()
        ]

    if sorted(order) != list(range(main_heads)):
        raise ValueError(
            f"fixed head order must be permutation of "
            f"0..{main_heads - 1}. Got {order}"
        )

    return order


# ============================================================
# Dataset
# ============================================================

def build_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(
                (args.img_size, args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.img_size, args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_set = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


# ============================================================
# Diversity loss
# ============================================================

def compute_diversity_loss(
    info_list,
    criterion,
):
    losses = []

    for info in info_list:
        losses.append(
            criterion(
                head_out=info["head_out"],
                active_mask=info["active_mask"],
                direct_mask=info["direct_mask"],
                mixed_mask=info["mixed_mask"],
            )
        )

    return torch.stack(losses).mean()


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    diversity_criterion,
    device,
    budgets,
    lambda_div,
):
    model.train()

    total_loss_sum = 0.0
    task_loss_sum = 0.0
    div_loss_sum = 0.0

    correct = 0
    total = 0

    allocator_grad_sum = 0.0
    allocator_grad_steps = 0

    budget_counter = Counter()

    budget_schedule = build_balanced_budget_schedule(
        len(loader),
        budgets,
    )

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        budget = budget_schedule[batch_idx]
        budget_counter[budget] += 1

        optimizer.zero_grad(set_to_none=True)

        logits, info_list = model(
            images,
            budget=budget,
            return_info=True,
        )

        task_loss = F.cross_entropy(
            logits,
            targets,
        )

        div_loss = compute_diversity_loss(
            info_list,
            diversity_criterion,
        )

        loss = task_loss + lambda_div * div_loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected. budget={budget}"
            )

        loss.backward()

        allocator_grad = (
            global_grad_norm_from_named_parameters(
                model.named_parameters(),
                name_contains="allocator",
            )
        )

        if budget > 0:
            if allocator_grad <= 0.0:
                raise RuntimeError(
                    "Allocator gradient became zero "
                    "during a budget>0 step."
                )

            allocator_grad_sum += allocator_grad
            allocator_grad_steps += 1

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

        optimizer.step()

        bs = targets.size(0)

        total_loss_sum += loss.item() * bs
        task_loss_sum += task_loss.item() * bs
        div_loss_sum += div_loss.item() * bs

        correct += (
            logits.argmax(dim=-1) == targets
        ).sum().item()

        total += bs

    return {
        "total_loss": total_loss_sum / total,
        "task_loss": task_loss_sum / total,
        "div_loss": div_loss_sum / total,
        "accuracy": 100.0 * correct / total,
        "allocator_grad_norm": (
            allocator_grad_sum / allocator_grad_steps
            if allocator_grad_steps > 0
            else 0.0
        ),
        "budget_batches": {
            str(b): int(budget_counter[b])
            for b in budgets
        },
    }


# ============================================================
# Allocation override for fixed / random ablation
# ============================================================

def _build_override_schedule(
    scheduler,
    alloc_logits,
    budget,
    order_idx,
):
    B, H = alloc_logits.shape

    direct_count = scheduler._get_direct_count(
        budget
    )

    active_mask = torch.zeros(
        B,
        H,
        dtype=torch.bool,
        device=alloc_logits.device,
    )

    direct_mask = torch.zeros_like(
        active_mask
    )

    if budget > 0:
        active_mask.scatter_(
            1,
            order_idx[:, :budget],
            True,
        )

    if direct_count > 0:
        direct_mask.scatter_(
            1,
            order_idx[:, :direct_count],
            True,
        )

    mixed_mask = active_mask & (~direct_mask)
    inactive_mask = ~active_mask

    dtype = alloc_logits.dtype

    active_gate = active_mask.to(dtype)
    direct_gate = direct_mask.to(dtype)
    mixed_gate = mixed_mask.to(dtype)
    inactive_gate = inactive_mask.to(dtype)

    rank_scores = torch.empty_like(
        alloc_logits
    )

    ranks = torch.arange(
        H,
        0,
        -1,
        dtype=dtype,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(B, -1)

    rank_scores.scatter_(
        1,
        order_idx,
        ranks,
    )

    stats = {
        "budget": torch.tensor(
            float(budget),
            device=alloc_logits.device,
        ),
        "active_count_mean": (
            active_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "direct_count_mean": (
            direct_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "mixed_count_mean": (
            mixed_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "inactive_count_mean": (
            inactive_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
    }

    return {
        "active_mask": active_mask,
        "direct_mask": direct_mask,
        "mixed_mask": mixed_mask,
        "inactive_mask": inactive_mask,
        "active_gate": active_gate,
        "direct_gate": direct_gate,
        "mixed_gate": mixed_gate,
        "inactive_gate": inactive_gate,
        "selection_scores": rank_scores,
        "gumbel_noise": torch.zeros_like(
            alloc_logits
        ),
        "stats": stats,
    }


@contextmanager
def temporary_allocation_mode(
    model,
    mode,
    fixed_head_order,
    random_seed,
):
    if mode == "learned":
        yield
        return

    original_forwards = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        scheduler = block.attn.scheduler

        original_forwards.append(
            (scheduler, scheduler.forward)
        )

        fixed_tensor = torch.tensor(
            fixed_head_order,
            dtype=torch.long,
        )

        generator = (
            torch.Generator(device="cpu")
            .manual_seed(
                random_seed
                + block_idx * 100003
            )
        )

        def override_forward(
            self_scheduler,
            alloc_logits,
            budget,
            *,
            _mode=mode,
            _fixed=fixed_tensor,
            _generator=generator,
        ):
            B, H = alloc_logits.shape

            if _mode == "fixed":
                order_idx = (
                    _fixed.to(
                        alloc_logits.device
                    )
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            else:
                scores = torch.rand(
                    B,
                    H,
                    generator=_generator,
                    device="cpu",
                )

                order_idx = (
                    scores.argsort(
                        dim=-1,
                        descending=True,
                    )
                    .to(
                        alloc_logits.device
                    )
                )

            return _build_override_schedule(
                self_scheduler,
                alloc_logits,
                budget,
                order_idx,
            )

        scheduler.forward = MethodType(
            override_forward,
            scheduler,
        )

    try:
        yield
    finally:
        for scheduler, original in (
            original_forwards
        ):
            scheduler.forward = original


# ============================================================
# Evaluation statistics
# ============================================================

def tensor_sample_norm(x):
    return (
        x.detach()
        .float()
        .flatten(start_dim=1)
        .norm(dim=1)
    )


class EvalStats:
    def __init__(self, main_heads):
        self.main_heads = main_heads

        self.active = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )

        self.direct = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )

        self.mixed = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )

        self.block_samples = 0

        self.logit_sum = 0.0
        self.logit_sq_sum = 0.0
        self.logit_numel = 0

        self.entropy_sum = 0.0
        self.entropy_count = 0

        self.patterns = Counter()

        self.mini_norm_sum = 0.0
        self.main_norm_sum = 0.0
        self.norm_count = 0

        self.attn_norm_sum = 0.0
        self.attn_norm_count = 0

    def update_info(self, info_list):
        for info in info_list:
            active = (
                info["active_mask"]
                .detach()
                .cpu()
            )

            direct = (
                info["direct_mask"]
                .detach()
                .cpu()
            )

            mixed = (
                info["mixed_mask"]
                .detach()
                .cpu()
            )

            logits = (
                info["alloc_logits"]
                .detach()
                .float()
                .cpu()
            )

            B = active.size(0)

            self.active += (
                active.float()
                .sum(dim=0)
                .double()
            )

            self.direct += (
                direct.float()
                .sum(dim=0)
                .double()
            )

            self.mixed += (
                mixed.float()
                .sum(dim=0)
                .double()
            )

            self.block_samples += B

            self.logit_sum += logits.sum().item()
            self.logit_sq_sum += (
                logits.pow(2).sum().item()
            )
            self.logit_numel += logits.numel()

            probs = logits.softmax(dim=-1)

            entropy = -(
                probs
                * (probs + 1e-8).log()
            ).sum(dim=-1)

            if self.main_heads > 1:
                entropy /= math.log(
                    self.main_heads
                )

            self.entropy_sum += entropy.sum().item()
            self.entropy_count += entropy.numel()

            for row in active.tolist():
                pattern = "".join(
                    "1" if x else "0"
                    for x in row
                )
                self.patterns[pattern] += 1

            mini_norm = tensor_sample_norm(
                info["mini_context"]
            )

            main_norm = tensor_sample_norm(
                info["main_out"]
            )

            self.mini_norm_sum += (
                mini_norm.sum().item()
            )

            self.main_norm_sum += (
                main_norm.sum().item()
            )

            self.norm_count += mini_norm.numel()

    def update_attn_output(self, output):
        if isinstance(output, tuple):
            output = output[0]

        norms = tensor_sample_norm(
            output
        )

        self.attn_norm_sum += (
            norms.sum().item()
        )

        self.attn_norm_count += (
            norms.numel()
        )

    def summary(self):
        denom = max(
            1,
            self.block_samples,
        )

        mean = (
            self.logit_sum / self.logit_numel
            if self.logit_numel > 0
            else 0.0
        )

        variance = (
            self.logit_sq_sum / self.logit_numel
            - mean * mean
            if self.logit_numel > 0
            else 0.0
        )

        logit_std = math.sqrt(
            max(variance, 0.0)
        )

        entropy = (
            self.entropy_sum
            / self.entropy_count
            if self.entropy_count > 0
            else 0.0
        )

        mini_norm = (
            self.mini_norm_sum
            / self.norm_count
            if self.norm_count > 0
            else 0.0
        )

        main_norm = (
            self.main_norm_sum
            / self.norm_count
            if self.norm_count > 0
            else 0.0
        )

        attn_norm = (
            self.attn_norm_sum
            / self.attn_norm_count
            if self.attn_norm_count > 0
            else 0.0
        )

        return {
            "active_head_freq_pct": (
                100.0 * self.active / denom
            ).tolist(),
            "direct_head_freq_pct": (
                100.0 * self.direct / denom
            ).tolist(),
            "mixed_head_freq_pct": (
                100.0 * self.mixed / denom
            ).tolist(),
            "alloc_logits_mean": mean,
            "alloc_logits_std": logit_std,
            "allocation_entropy_norm": entropy,
            "unique_active_patterns": len(
                self.patterns
            ),
            "active_pattern_top5": (
                self.patterns.most_common(5)
            ),
            "mini_context_norm": mini_norm,
            "main_out_norm": main_norm,
            "attn_out_norm": attn_norm,
            "main_to_mini_norm_ratio": (
                main_norm / mini_norm
                if mini_norm > 0
                else 0.0
            ),
        }


class HookGroup:
    def __init__(self, model, stats):
        self.handles = []

        for block in model.blocks:
            handle = (
                block.attn.register_forward_hook(
                    lambda module, inputs, output,
                    _stats=stats:
                    _stats.update_attn_output(
                        output
                    )
                )
            )

            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.remove()


@torch.no_grad()
def evaluate_budget(
    model,
    loader,
    device,
    budget,
    mode,
    fixed_head_order,
    random_seed,
):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    stats = EvalStats(
        model.main_heads
    )

    with temporary_allocation_mode(
        model,
        mode,
        fixed_head_order,
        random_seed,
    ):
        with HookGroup(model, stats):
            for images, targets in loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                targets = targets.to(
                    device,
                    non_blocking=True,
                )

                logits, info_list = model(
                    images,
                    budget=budget,
                    return_info=True,
                )

                loss = F.cross_entropy(
                    logits,
                    targets,
                    reduction="sum",
                )

                correct += (
                    logits.argmax(dim=-1)
                    == targets
                ).sum().item()

                total += targets.size(0)
                loss_sum += loss.item()

                stats.update_info(
                    info_list
                )

    result = {
        "accuracy": 100.0 * correct / total,
        "loss": loss_sum / total,
        "allocation_mode": mode,
        "budget": budget,
    }

    result.update(
        stats.summary()
    )

    return result


@torch.no_grad()
def evaluate_all(
    model,
    loader,
    device,
    budgets,
    fixed_head_order,
    random_seed,
):
    results = {
        "learned": {},
        "fixed": {},
        "random": {},
    }

    # learned
    for budget in budgets:
        results["learned"][str(budget)] = (
            evaluate_budget(
                model,
                loader,
                device,
                budget,
                "learned",
                fixed_head_order,
                random_seed + budget,
            )
        )

    # fixed / random
    for mode_idx, mode in enumerate(
        ["fixed", "random"]
    ):
        for budget in budgets:
            if budget == 0:
                copied = dict(
                    results["learned"]["0"]
                )
                copied["allocation_mode"] = mode
                results[mode]["0"] = copied
                continue

            results[mode][str(budget)] = (
                evaluate_budget(
                    model,
                    loader,
                    device,
                    budget,
                    mode,
                    fixed_head_order,
                    (
                        random_seed
                        + mode_idx * 100003
                        + budget
                    ),
                )
            )

    return results


# ============================================================
# Diagnostics
# ============================================================

def detect_head_collapse(
    learned_result,
    threshold_pct=90.0,
):
    freq = learned_result[
        "active_head_freq_pct"
    ]

    return (
        max(freq) >= threshold_pct
        if freq
        else False
    )


def budget_monotonic_score(
    learned_results,
):
    """
    0~1.
    인접 budget accuracy가 non-decreasing한 비율.
    """
    budgets = sorted(
        int(k)
        for k in learned_results.keys()
    )

    acc = [
        learned_results[str(b)][
            "accuracy"
        ]
        for b in budgets
    ]

    good = 0

    for i in range(len(acc) - 1):
        if acc[i + 1] >= acc[i]:
            good += 1

    return (
        good / (len(acc) - 1)
        if len(acc) > 1
        else 1.0
    )


def print_epoch_report(
    epoch,
    epochs,
    tau,
    lr,
    train_stats,
    eval_stats,
    elapsed,
):
    print()
    print("=" * 108)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | "
        f"tau={tau:.4f} | "
        f"lr={lr:.3e} | "
        f"time={elapsed:.1f}s"
    )

    print(
        f"Train | "
        f"loss={train_stats['total_loss']:.4f} "
        f"task={train_stats['task_loss']:.4f} "
        f"div={train_stats['div_loss']:.4f} "
        f"acc={train_stats['accuracy']:.2f}% "
        f"allocator_grad="
        f"{train_stats['allocator_grad_norm']:.6e}"
    )

    print(
        "Budget batches:",
        train_stats["budget_batches"],
    )

    for budget in sorted(
        int(x)
        for x in eval_stats["learned"]
    ):
        b = str(budget)

        print()
        print(f"B={budget}")

        for mode in [
            "learned",
            "fixed",
            "random",
        ]:
            r = eval_stats[mode][b]

            print(
                f"  {mode:<7} "
                f"acc={r['accuracy']:6.2f}% "
                f"loss={r['loss']:.4f} | "
                f"mini={r['mini_context_norm']:.2f} "
                f"main={r['main_out_norm']:.2f} "
                f"attn={r['attn_out_norm']:.2f} "
                f"main/mini="
                f"{r['main_to_mini_norm_ratio']:.3f}"
            )

        learned = eval_stats[
            "learned"
        ][b]

        print(
            "  learned active:",
            " ".join(
                f"H{i}:{v:5.1f}%"
                for i, v in enumerate(
                    learned[
                        "active_head_freq_pct"
                    ]
                )
            ),
        )

        print(
            f"  allocator: "
            f"std="
            f"{learned['alloc_logits_std']:.4f} "
            f"entropy="
            f"{learned['allocation_entropy_norm']:.4f} "
            f"patterns="
            f"{learned['unique_active_patterns']}"
        )

        if budget > 0:
            learned_acc = (
                eval_stats["learned"][b][
                    "accuracy"
                ]
            )
            fixed_acc = (
                eval_stats["fixed"][b][
                    "accuracy"
                ]
            )
            random_acc = (
                eval_stats["random"][b][
                    "accuracy"
                ]
            )

            print(
                f"  delta: "
                f"learned-fixed="
                f"{learned_acc-fixed_acc:+.2f}pp | "
                f"learned-random="
                f"{learned_acc-random_acc:+.2f}pp"
            )

    monotonic = budget_monotonic_score(
        eval_stats["learned"]
    )

    print()
    print(
        f"Budget monotonic score: "
        f"{monotonic:.2f}"
    )

    # B=1 collapse warning
    if "1" in eval_stats["learned"]:
        if detect_head_collapse(
            eval_stats["learned"]["1"]
        ):
            print(
                "WARNING: B=1 head selection "
                "looks collapsed (>90% one head)."
            )


# ============================================================
# Main
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="./datasets",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/cifar10_sanity",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--img-size",
        type=int,
        default=224,
    )

    p.add_argument(
        "--patch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--embed-dim",
        type=int,
        default=192,
    )

    p.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    p.add_argument(
        "--main-heads",
        type=int,
        default=3,
    )

    p.add_argument(
        "--mini-heads",
        type=int,
        default=1,
    )

    p.add_argument(
        "--mini-dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--pool-ratio",
        type=int,
        default=2,
    )

    p.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
    )

    p.add_argument(
        "--direct-ratio",
        type=float,
        default=0.34,
    )

    p.add_argument(
        "--alpha-direct",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--alpha-mixed",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--lambda-div",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--tau-start",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--tau-end",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--fixed-head-order",
        type=str,
        default="",
    )

    p.add_argument(
        "--random-eval-seed",
        type=int,
        default=2026,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    return p


def main():
    global args

    args = build_parser().parse_args()

    set_seed(args.seed)

    device = resolve_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    budgets = list(
        range(args.main_heads + 1)
    )

    fixed_head_order = (
        parse_fixed_head_order(
            args.fixed_head_order,
            args.main_heads,
        )
    )

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print()
    print(
        "Strong sanity config:"
    )
    print(
        f"  full CIFAR-10 | "
        f"epochs={args.epochs} | "
        f"batch={args.batch_size}"
    )
    print(
        f"  img={args.img_size}, "
        f"patch={args.patch_size}, "
        f"embed={args.embed_dim}, "
        f"depth={args.depth}"
    )
    print(
        f"  main_heads={args.main_heads}, "
        f"mini_heads={args.mini_heads}, "
        f"budgets={budgets}"
    )
    print(
        f"  fixed order={fixed_head_order}"
    )

    train_loader, test_loader = (
        build_loaders(args)
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    model = MiniGuidedViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=10,
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=128,
        gumbel_tau=args.tau_start,
        use_gumbel=True,
    ).to(device)

    diversity_criterion = (
        HeadDiversityLoss(
            mode="direct_mixed"
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    metrics_path = (
        output_dir / "metrics.jsonl"
    )

    best_avg_acc = -1.0

    for epoch_idx in range(
        args.epochs
    ):
        start = time.time()

        epoch = epoch_idx + 1

        tau = gumbel_temperature(
            epoch_idx,
            args.epochs,
            args.tau_start,
            args.tau_end,
        )

        model.set_gumbel_temperature(
            tau
        )

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            diversity_criterion,
            device,
            budgets,
            args.lambda_div,
        )

        eval_stats = evaluate_all(
            model,
            test_loader,
            device,
            budgets,
            fixed_head_order,
            args.random_eval_seed,
        )

        elapsed = time.time() - start

        lr = optimizer.param_groups[0][
            "lr"
        ]

        print_epoch_report(
            epoch,
            args.epochs,
            tau,
            lr,
            train_stats,
            eval_stats,
            elapsed,
        )

        learned_avg_acc = sum(
            r["accuracy"]
            for r in eval_stats[
                "learned"
            ].values()
        ) / len(
            eval_stats["learned"]
        )

        record = {
            "epoch": epoch,
            "tau": tau,
            "lr": lr,
            "elapsed_sec": elapsed,
            "train": train_stats,
            "eval": eval_stats,
            "learned_average_budget_accuracy": (
                learned_avg_acc
            ),
            "budget_monotonic_score": (
                budget_monotonic_score(
                    eval_stats["learned"]
                )
            ),
        }

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": (
                optimizer.state_dict()
            ),
            "scheduler": (
                lr_scheduler.state_dict()
            ),
            "tau": tau,
            "eval": eval_stats,
            "args": vars(args),
        }

        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        if learned_avg_acc > best_avg_acc:
            best_avg_acc = learned_avg_acc

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

        lr_scheduler.step()

    print()
    print("=" * 108)
    print(
        "Strong sanity finished."
    )
    print(
        f"Best learned average budget "
        f"accuracy: {best_avg_acc:.2f}%"
    )
    print(
        f"Metrics: {metrics_path}"
    )
    print(
        f"Best checkpoint: "
        f"{output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()