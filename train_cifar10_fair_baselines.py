# train_cifar10_fair_baselines.py
"""
Fair separately-trained baseline experiment for Mini-to-Main Attention.

Goal
----
Compare THREE models trained FROM SCRATCH under the same protocol:

    1) learned : input-dependent Gumbel-ST allocator
    2) fixed   : fixed nested head order in every block
    3) random  : input-independent random head order per sample/block

Data protocol
-------------
CIFAR-10 train 50,000:
    train split      45,000
    validation split  5,000

CIFAR-10 official test 10,000:
    NEVER used for checkpoint selection.
    Evaluated only after all three best checkpoints are finalized.

Model/training
--------------
Default:
    img_size=224
    patch_size=16
    embed_dim=192
    depth=6
    main_heads=3
    mini_heads=1
    budgets=[0,1,2,3]
    epochs=50
    batch_size=128
    CE + 0.01 * direct-mixed diversity
    AdamW(lr=3e-4, wd=0.05)
    cosine LR
    Gumbel tau 1.5 -> 0.5 for learned model

Fairness
--------
- Same train/validation indices for all modes.
- Same model initialization seed for all modes.
- Same DataLoader shuffle seed for all modes.
- Same optimizer / LR / budget schedule.
- Fixed/random allocator parameters are frozen because their schedulers
  deliberately ignore alloc_logits.
- No official test-set evaluation occurs during training.

Why fixed order is NOT copied from a previously trained learned model
---------------------------------------------------------------------
When training the fixed baseline from scratch, H0/H1/H2 identities are
permutation-symmetric at initialization. A "best H2 from another checkpoint"
has no principled identity correspondence after reinitialization.

Therefore fixed uses a canonical per-block ordering:
    [H0, H1, H2]

With main_heads=3 this means:
    B=1 -> H0 direct
    B=2 -> H0 direct + H1 mixed
    B=3 -> all active; H0 remains direct under direct_ratio=0.34

The fixed model is given 50 epochs to adapt its weights to this policy.

Random baseline
---------------
During training, each sample and block receives a random permutation of
the three heads. During validation, a fixed random seed is used so
checkpoint selection is reproducible. Final test reports multiple random
routing repeats (default 5) as mean/std.

Output
------
output_dir/
    split_indices.json
    run_config.json
    learned/
        metrics.jsonl
        last.pt
        best.pt
    fixed/
        metrics.jsonl
        last.pt
        best.pt
    random/
        metrics.jsonl
        last.pt
        best.pt
    final_test.json

IMPORTANT
---------
Current v1 still computes all Main heads densely and gates afterward.
This experiment tests ACCURACY / routing utility, not actual FLOP savings.
"""

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import train_cifar10_sanity as base
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


def parse_modes(text: str) -> List[str]:
    modes = [
        x.strip().lower()
        for x in text.split(",")
        if x.strip()
    ]

    allowed = {"learned", "fixed", "random"}

    if not modes:
        raise ValueError("--modes cannot be empty.")

    if len(set(modes)) != len(modes):
        raise ValueError(
            f"Duplicate modes in --modes: {modes}"
        )

    bad = [
        x for x in modes
        if x not in allowed
    ]

    if bad:
        raise ValueError(
            f"Unknown modes {bad}. Allowed: {sorted(allowed)}"
        )

    return modes


def parse_fixed_order(
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
            "--fixed-order must be a permutation of "
            f"0..{main_heads - 1}; got {order}."
        )

    return order


# ============================================================
# Dataset split / loaders
# ============================================================

def build_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(
                (img_size, img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    return (
        train_transform,
        eval_transform,
    )


def make_split_indices(
    num_samples: int,
    val_size: int,
    split_seed: int,
):
    if val_size <= 0 or val_size >= num_samples:
        raise ValueError(
            f"val_size must be in [1,{num_samples-1}], "
            f"got {val_size}."
        )

    g = torch.Generator()
    g.manual_seed(split_seed)

    perm = torch.randperm(
        num_samples,
        generator=g,
    ).tolist()

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    return (
        train_indices,
        val_indices,
    )


def build_datasets(
    data_dir: str,
    img_size: int,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
):
    train_transform, eval_transform = (
        build_transforms(img_size)
    )

    # Separate dataset objects are intentional:
    # train subset uses augmentation while validation does not.
    full_train_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    full_train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_set = Subset(
        full_train_aug,
        list(train_indices),
    )

    val_set = Subset(
        full_train_eval,
        list(val_indices),
    )

    return (
        train_set,
        val_set,
        test_set,
    )


def build_train_loader(
    train_set,
    batch_size: int,
    num_workers: int,
    loader_seed: int,
):
    g = torch.Generator()
    g.manual_seed(loader_seed)

    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_eval_loader(
    dataset,
    batch_size: int,
    num_workers: int,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# Model
# ============================================================

def build_model(
    args,
    device: torch.device,
) -> MiniGuidedViT:
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

    return model


def freeze_allocator_for_nonlearned(
    model,
    mode: str,
):
    if mode == "learned":
        return

    for name, p in model.named_parameters():
        if "allocator" in name:
            p.requires_grad_(False)


def trainable_parameter_count(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Fixed / random scheduler override
# ============================================================

def _build_override_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    order_idx: torch.Tensor,
):
    B, H = alloc_logits.shape

    if order_idx.shape != (B, H):
        raise ValueError(
            f"order_idx shape {tuple(order_idx.shape)} "
            f"!= {(B,H)}"
        )

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

    mixed_mask = (
        active_mask & (~direct_mask)
    )

    inactive_mask = ~active_mask

    dtype = alloc_logits.dtype

    # Rank scores are only diagnostic here.
    rank_scores = torch.zeros_like(
        alloc_logits
    )

    ranks = torch.arange(
        H,
        0,
        -1,
        dtype=dtype,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(
        B,
        -1,
    )

    rank_scores.scatter_(
        1,
        order_idx,
        ranks,
    )

    return {
        "active_mask": active_mask,
        "direct_mask": direct_mask,
        "mixed_mask": mixed_mask,
        "inactive_mask": inactive_mask,
        "active_gate": active_mask.to(dtype),
        "direct_gate": direct_mask.to(dtype),
        "mixed_gate": mixed_mask.to(dtype),
        "inactive_gate": inactive_mask.to(dtype),
        "selection_scores": rank_scores,
        "gumbel_noise": torch.zeros_like(
            alloc_logits
        ),
        "stats": {
            "budget": torch.tensor(
                float(budget),
                device=alloc_logits.device,
                dtype=dtype,
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
        },
    }


@contextmanager
def allocation_mode(
    model,
    mode: str,
    fixed_order: Sequence[int],
    random_seed: int,
):
    """
    learned:
        original differentiable scheduler

    fixed:
        same fixed full ordering in every block and every sample

    random:
        random full ordering per sample/block
        using deterministic CPU generators initialized by random_seed
    """
    if mode == "learned":
        yield
        return

    if mode not in {"fixed", "random"}:
        raise ValueError(
            f"Unsupported allocation mode: {mode}"
        )

    originals = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        scheduler = block.attn.scheduler

        originals.append(
            (scheduler, scheduler.forward)
        )

        fixed_tensor = torch.tensor(
            list(fixed_order),
            dtype=torch.long,
        )

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            random_seed
            + block_idx * 100003
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
                    _fixed
                    .to(
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
                scheduler=self_scheduler,
                alloc_logits=alloc_logits,
                budget=budget,
                order_idx=order_idx,
            )

        scheduler.forward = MethodType(
            override_forward,
            scheduler,
        )

    try:
        yield
    finally:
        for scheduler, original in originals:
            scheduler.forward = original


# ============================================================
# Train
# ============================================================

def deterministic_balanced_budget_schedule(
    num_batches: int,
    budgets: Sequence[int],
    seed: int,
) -> List[int]:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    if not budgets:
        raise ValueError("budgets cannot be empty.")

    repeats = math.ceil(
        num_batches / len(budgets)
    )

    schedule = (
        list(budgets) * repeats
    )[:num_batches]

    rng = random.Random(seed)
    rng.shuffle(schedule)

    return schedule


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

    return torch.stack(
        losses
    ).mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    diversity_criterion,
    device,
    budgets,
    lambda_div,
    grad_clip,
    mode,
    fixed_order,
    routing_seed,
    budget_seed,
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

    budget_schedule = (
        deterministic_balanced_budget_schedule(
            num_batches=len(loader),
            budgets=budgets,
            seed=budget_seed,
        )
    )

    with allocation_mode(
        model=model,
        mode=mode,
        fixed_order=fixed_order,
        random_seed=routing_seed,
    ):
        for batch_idx, (
            images,
            targets,
        ) in enumerate(loader):
            images = images.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            budget = (
                budget_schedule[
                    batch_idx
                ]
            )

            budget_counter[
                budget
            ] += 1

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, info_list = model(
                images,
                budget=budget,
                return_info=True,
            )

            task_loss = F.cross_entropy(
                logits,
                targets,
            )

            div_loss = (
                compute_diversity_loss(
                    info_list,
                    diversity_criterion,
                )
            )

            loss = (
                task_loss
                + lambda_div * div_loss
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss: "
                    f"mode={mode}, budget={budget}"
                )

            loss.backward()

            if mode == "learned":
                allocator_grad = (
                    base.global_grad_norm_from_named_parameters(
                        model.named_parameters(),
                        name_contains="allocator",
                    )
                )

                if budget > 0:
                    if allocator_grad <= 0.0:
                        raise RuntimeError(
                            "Learned allocator gradient "
                            "became zero on a budget>0 step."
                        )

                    allocator_grad_sum += (
                        allocator_grad
                    )

                    allocator_grad_steps += 1

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (
                        p
                        for p in model.parameters()
                        if p.requires_grad
                    ),
                    max_norm=grad_clip,
                )

            optimizer.step()

            bs = targets.size(0)

            total_loss_sum += (
                loss.item() * bs
            )

            task_loss_sum += (
                task_loss.item() * bs
            )

            div_loss_sum += (
                div_loss.item() * bs
            )

            correct += (
                logits.argmax(dim=-1)
                == targets
            ).sum().item()

            total += bs

    return {
        "total_loss": (
            total_loss_sum / total
        ),
        "task_loss": (
            task_loss_sum / total
        ),
        "div_loss": (
            div_loss_sum / total
        ),
        "accuracy": (
            100.0 * correct / total
        ),
        "allocator_grad_norm": (
            allocator_grad_sum
            / allocator_grad_steps
            if allocator_grad_steps > 0
            else 0.0
        ),
        "budget_batches": {
            str(b): int(
                budget_counter[b]
            )
            for b in budgets
        },
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_budget(
    model,
    loader,
    device,
    budget,
    mode,
    fixed_order,
    random_seed,
    collect_correctness=False,
):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    correctness = []
    targets_all = []

    with allocation_mode(
        model=model,
        mode=mode,
        fixed_order=fixed_order,
        random_seed=random_seed,
    ):
        for images, targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images,
                budget=budget,
                return_info=False,
            )

            if isinstance(
                logits,
                tuple,
            ):
                logits = logits[0]

            loss = F.cross_entropy(
                logits,
                targets,
                reduction="sum",
            )

            pred = logits.argmax(
                dim=-1
            )

            batch_correct = (
                pred == targets
            )

            correct += (
                batch_correct
                .sum()
                .item()
            )

            total += (
                targets.size(0)
            )

            loss_sum += (
                loss.item()
            )

            if collect_correctness:
                correctness.append(
                    batch_correct
                    .detach()
                    .cpu()
                )

                targets_all.append(
                    targets
                    .detach()
                    .cpu()
                )

    result = {
        "budget": budget,
        "mode": mode,
        "accuracy": (
            100.0 * correct / total
        ),
        "loss": (
            loss_sum / total
        ),
        "samples": total,
    }

    if collect_correctness:
        result["correctness"] = (
            torch.cat(
                correctness
            ).bool()
        )

        result["targets"] = (
            torch.cat(
                targets_all
            )
        )

    return result


@torch.no_grad()
def evaluate_all_budgets(
    model,
    loader,
    device,
    budgets,
    mode,
    fixed_order,
    random_seed,
):
    results = {}

    for budget in budgets:
        results[str(budget)] = (
            evaluate_budget(
                model=model,
                loader=loader,
                device=device,
                budget=budget,
                mode=mode,
                fixed_order=fixed_order,
                random_seed=(
                    random_seed
                    + budget * 1009
                ),
                collect_correctness=False,
            )
        )

    return results


def average_budget_accuracy(
    results,
):
    return sum(
        item["accuracy"]
        for item in results.values()
    ) / len(results)


def budget_monotonic_score(
    results,
):
    budgets = sorted(
        int(k)
        for k in results.keys()
    )

    acc = [
        results[str(b)][
            "accuracy"
        ]
        for b in budgets
    ]

    good = sum(
        1
        for i in range(
            len(acc) - 1
        )
        if acc[i + 1] >= acc[i]
    )

    if len(acc) <= 1:
        return 1.0

    return (
        good
        / (len(acc) - 1)
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    *,
    mode,
    epoch,
    model,
    optimizer,
    lr_scheduler,
    tau,
    val_results,
    val_avg,
    best_val_avg,
    fixed_order,
    args,
):
    torch.save(
        {
            "mode": mode,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": (
                optimizer.state_dict()
            ),
            "scheduler": (
                lr_scheduler.state_dict()
            ),
            "tau": tau,
            "val": val_results,
            "val_average_budget_accuracy": (
                val_avg
            ),
            "best_val_average_budget_accuracy": (
                best_val_avg
            ),
            "fixed_order": list(
                fixed_order
            ),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint_model(
    checkpoint_path: Path,
    args,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        args,
        device,
    )

    mode = checkpoint.get(
        "mode",
        "learned",
    )

    freeze_allocator_for_nonlearned(
        model,
        mode,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# Per-mode training
# ============================================================

def print_epoch(
    mode,
    epoch,
    epochs,
    tau,
    lr,
    train_stats,
    val_results,
    val_avg,
    elapsed,
):
    print()
    print("=" * 112)

    print(
        f"[{mode.upper()}] "
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
        train_stats[
            "budget_batches"
        ],
    )

    for b in sorted(
        int(x)
        for x in val_results
    ):
        item = val_results[
            str(b)
        ]

        print(
            f"  Val B={b}: "
            f"acc={item['accuracy']:.2f}% "
            f"loss={item['loss']:.4f}"
        )

    print(
        f"Val average budget accuracy: "
        f"{val_avg:.2f}%"
    )

    print(
        f"Val budget monotonic score: "
        f"{budget_monotonic_score(val_results):.2f}"
    )


def train_mode(
    *,
    mode,
    args,
    device,
    train_set,
    val_loader,
    fixed_order,
    budgets,
):
    mode_dir = (
        Path(args.output_dir)
        / mode
    )

    mode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same model initialization for each baseline.
    set_seed(
        args.seed
    )

    model = build_model(
        args,
        device,
    )

    freeze_allocator_for_nonlearned(
        model,
        mode,
    )

    print()
    print("#" * 112)
    print(
        f"TRAIN MODE: {mode.upper()}"
    )
    print("#" * 112)

    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count(model):,}"
    )

    if mode == "fixed":
        print(
            f"Fixed head order in every block: "
            f"{fixed_order}"
        )

    diversity_criterion = (
        HeadDiversityLoss(
            mode="direct_mixed"
        )
    )

    optimizer = torch.optim.AdamW(
        (
            p
            for p in model.parameters()
            if p.requires_grad
        ),
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
        mode_dir
        / "metrics.jsonl"
    )

    best_val_avg = -float(
        "inf"
    )

    start_epoch = 0

    last_path = (
        mode_dir / "last.pt"
    )

    if (
        args.resume_existing
        and last_path.exists()
    ):
        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        lr_scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        )

        best_val_avg = float(
            checkpoint[
                "best_val_average_budget_accuracy"
            ]
        )

        print(
            f"Resuming {mode} from epoch "
            f"{start_epoch}."
        )

    if start_epoch >= args.epochs:
        print(
            f"{mode} already completed "
            f"{start_epoch} epochs."
        )

        return (
            mode_dir / "best.pt"
        )

    for epoch_idx in range(
        start_epoch,
        args.epochs,
    ):
        epoch = epoch_idx + 1

        start_time = time.time()

        tau = base.gumbel_temperature(
            epoch_idx=epoch_idx,
            epochs=args.epochs,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
        )

        model.set_gumbel_temperature(
            tau
        )

        # Epoch-specific loader seed:
        # identical across modes and stable across resume.
        train_loader = build_train_loader(
            train_set=train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            loader_seed=(
                args.loader_seed
                + epoch_idx
            ),
        )

        # Epoch-specific balanced-budget seed:
        # identical across modes and stable across resume.
        budget_seed = (
            args.seed * 1000003
            + epoch * 7919
            + 17
        )

        # For random-trained routing, vary routing seed by epoch.
        train_routing_seed = (
            args.random_train_seed
            + args.seed * 1000003
            + epoch * 10007
        )

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            diversity_criterion=(
                diversity_criterion
            ),
            device=device,
            budgets=budgets,
            lambda_div=args.lambda_div,
            grad_clip=args.grad_clip,
            mode=mode,
            fixed_order=fixed_order,
            routing_seed=(
                train_routing_seed
            ),
            budget_seed=(
                budget_seed
            ),
        )

        # Validation is the ONLY checkpoint-selection signal.
        # Random validation uses the same deterministic seed every epoch.
        val_results = (
            evaluate_all_budgets(
                model=model,
                loader=val_loader,
                device=device,
                budgets=budgets,
                mode=mode,
                fixed_order=fixed_order,
                random_seed=(
                    args.random_val_seed
                ),
            )
        )

        val_avg = (
            average_budget_accuracy(
                val_results
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print_epoch(
            mode=mode,
            epoch=epoch,
            epochs=args.epochs,
            tau=tau,
            lr=lr,
            train_stats=(
                train_stats
            ),
            val_results=(
                val_results
            ),
            val_avg=val_avg,
            elapsed=elapsed,
        )

        record = {
            "mode": mode,
            "epoch": epoch,
            "tau": tau,
            "lr": lr,
            "elapsed_sec": elapsed,
            "train": train_stats,
            "val": val_results,
            "val_average_budget_accuracy": (
                val_avg
            ),
            "val_budget_monotonic_score": (
                budget_monotonic_score(
                    val_results
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

        improved = (
            val_avg
            > best_val_avg
        )

        if improved:
            best_val_avg = val_avg

        # Step before saving so resume aligns with uninterrupted run.
        lr_scheduler.step()

        save_checkpoint(
            last_path,
            mode=mode,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            lr_scheduler=(
                lr_scheduler
            ),
            tau=tau,
            val_results=(
                val_results
            ),
            val_avg=val_avg,
            best_val_avg=(
                best_val_avg
            ),
            fixed_order=fixed_order,
            args=args,
        )

        if improved:
            best_path = (
                mode_dir / "best.pt"
            )

            save_checkpoint(
                best_path,
                mode=mode,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=(
                    lr_scheduler
                ),
                tau=tau,
                val_results=(
                    val_results
                ),
                val_avg=val_avg,
                best_val_avg=(
                    best_val_avg
                ),
                fixed_order=(
                    fixed_order
                ),
                args=args,
            )

            print(
                f"New {mode} best: "
                f"val avg={best_val_avg:.2f}%"
            )

    print()
    print(
        f"{mode.upper()} finished. "
        f"Best val average="
        f"{best_val_avg:.2f}%"
    )

    return (
        mode_dir / "best.pt"
    )


# ============================================================
# Paired McNemar learned vs fixed
# ============================================================

def paired_correctness(
    learned_correct,
    fixed_correct,
):
    learned_correct = (
        learned_correct.bool()
    )

    fixed_correct = (
        fixed_correct.bool()
    )

    if (
        learned_correct.shape
        != fixed_correct.shape
    ):
        raise ValueError(
            "Correctness shape mismatch."
        )

    both_correct = int(
        (
            learned_correct
            & fixed_correct
        ).sum().item()
    )

    both_wrong = int(
        (
            (~learned_correct)
            & (~fixed_correct)
        ).sum().item()
    )

    learned_only = int(
        (
            learned_correct
            & (~fixed_correct)
        ).sum().item()
    )

    fixed_only = int(
        (
            (~learned_correct)
            & fixed_correct
        ).sum().item()
    )

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "learned_only_correct": (
            learned_only
        ),
        "fixed_only_correct": (
            fixed_only
        ),
        "discordant_total": (
            learned_only
            + fixed_only
        ),
    }


def _log_binomial_pmf_half(
    n,
    k,
):
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )


def _logsumexp(values):
    m = max(values)

    return (
        m
        + math.log(
            sum(
                math.exp(v - m)
                for v in values
            )
        )
    )


def exact_mcnemar(
    learned_only,
    fixed_only,
):
    b = int(
        learned_only
    )

    c = int(
        fixed_only
    )

    n = b + c

    if n == 0:
        return {
            "exact_two_sided_p": 1.0,
            "chi2_continuity_corrected": 0.0,
            "chi2_approx_p": 1.0,
        }

    k = min(
        b,
        c,
    )

    log_lower = _logsumexp(
        [
            _log_binomial_pmf_half(
                n,
                i,
            )
            for i in range(
                k + 1
            )
        ]
    )

    exact_p = min(
        1.0,
        2.0
        * math.exp(
            log_lower
        ),
    )

    chi2 = (
        max(
            abs(b - c) - 1,
            0,
        )
        ** 2
    ) / n

    approx_p = math.erfc(
        math.sqrt(
            chi2 / 2.0
        )
    )

    return {
        "exact_two_sided_p": (
            exact_p
        ),
        "chi2_continuity_corrected": (
            chi2
        ),
        "chi2_approx_p": (
            approx_p
        ),
    }


# ============================================================
# Final untouched test
# ============================================================

def deterministic_test_model(
    *,
    checkpoint_path,
    mode,
    args,
    device,
    test_loader,
    budgets,
    fixed_order,
):
    model, checkpoint = (
        load_checkpoint_model(
            checkpoint_path,
            args,
            device,
        )
    )

    results = {}

    for budget in budgets:
        results[str(budget)] = (
            evaluate_budget(
                model=model,
                loader=test_loader,
                device=device,
                budget=budget,
                mode=mode,
                fixed_order=fixed_order,
                random_seed=(
                    args.random_test_seed
                    + budget * 1009
                ),
                collect_correctness=True,
            )
        )

    return (
        model,
        checkpoint,
        results,
    )


def random_test_model(
    *,
    checkpoint_path,
    args,
    device,
    test_loader,
    budgets,
    fixed_order,
):
    model, checkpoint = (
        load_checkpoint_model(
            checkpoint_path,
            args,
            device,
        )
    )

    repeats = []

    for repeat_idx in range(
        args.random_test_repeats
    ):
        seed = (
            args.random_test_seed
            + repeat_idx
            * 1000003
        )

        result = {}

        for budget in budgets:
            item = evaluate_budget(
                model=model,
                loader=test_loader,
                device=device,
                budget=budget,
                mode="random",
                fixed_order=fixed_order,
                random_seed=(
                    seed
                    + budget * 1009
                ),
                collect_correctness=False,
            )

            result[
                str(budget)
            ] = item

        repeats.append(
            {
                "repeat": (
                    repeat_idx
                ),
                "seed": seed,
                "budgets": result,
                "average_budget_accuracy": (
                    average_budget_accuracy(
                        result
                    )
                ),
            }
        )

    summary = {}

    for budget in budgets:
        b = str(budget)

        accs = [
            item["budgets"][b][
                "accuracy"
            ]
            for item in repeats
        ]

        losses = [
            item["budgets"][b][
                "loss"
            ]
            for item in repeats
        ]

        summary[b] = {
            "accuracy_mean": (
                statistics.mean(
                    accs
                )
            ),
            "accuracy_std": (
                statistics.stdev(
                    accs
                )
                if len(accs) > 1
                else 0.0
            ),
            "loss_mean": (
                statistics.mean(
                    losses
                )
            ),
            "loss_std": (
                statistics.stdev(
                    losses
                )
                if len(losses) > 1
                else 0.0
            ),
        }

    avg_accs = [
        item[
            "average_budget_accuracy"
        ]
        for item in repeats
    ]

    return (
        model,
        checkpoint,
        {
            "repeats": repeats,
            "summary": summary,
            "average_budget_accuracy_mean": (
                statistics.mean(
                    avg_accs
                )
            ),
            "average_budget_accuracy_std": (
                statistics.stdev(
                    avg_accs
                )
                if len(avg_accs) > 1
                else 0.0
            ),
        },
    )


def strip_tensor_fields(
    result,
):
    return {
        k: v
        for k, v in result.items()
        if k not in {
            "correctness",
            "targets",
        }
    }


def run_final_test(
    *,
    args,
    device,
    test_loader,
    budgets,
    fixed_order,
):
    root = Path(
        args.output_dir
    )

    required = {
        "learned": (
            root
            / "learned"
            / "best.pt"
        ),
        "fixed": (
            root
            / "fixed"
            / "best.pt"
        ),
        "random": (
            root
            / "random"
            / "best.pt"
        ),
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Cannot run final test; missing checkpoints:\n"
            + "\n".join(missing)
        )

    print()
    print("#" * 112)
    print(
        "FINAL UNTOUCHED CIFAR-10 TEST"
    )
    print("#" * 112)

    (
        learned_model,
        learned_ckpt,
        learned_results,
    ) = deterministic_test_model(
        checkpoint_path=(
            required["learned"]
        ),
        mode="learned",
        args=args,
        device=device,
        test_loader=test_loader,
        budgets=budgets,
        fixed_order=fixed_order,
    )

    (
        fixed_model,
        fixed_ckpt,
        fixed_results,
    ) = deterministic_test_model(
        checkpoint_path=(
            required["fixed"]
        ),
        mode="fixed",
        args=args,
        device=device,
        test_loader=test_loader,
        budgets=budgets,
        fixed_order=fixed_order,
    )

    (
        _,
        random_ckpt,
        random_result,
    ) = random_test_model(
        checkpoint_path=(
            required["random"]
        ),
        args=args,
        device=device,
        test_loader=test_loader,
        budgets=budgets,
        fixed_order=fixed_order,
    )

    learned_avg = (
        average_budget_accuracy(
            learned_results
        )
    )

    fixed_avg = (
        average_budget_accuracy(
            fixed_results
        )
    )

    print()
    print(
        f"Learned best epoch: "
        f"{learned_ckpt['epoch']} | "
        f"val avg="
        f"{learned_ckpt['best_val_average_budget_accuracy']:.2f}%"
    )

    print(
        f"Fixed best epoch: "
        f"{fixed_ckpt['epoch']} | "
        f"val avg="
        f"{fixed_ckpt['best_val_average_budget_accuracy']:.2f}%"
    )

    print(
        f"Random best epoch: "
        f"{random_ckpt['epoch']} | "
        f"val avg="
        f"{random_ckpt['best_val_average_budget_accuracy']:.2f}%"
    )

    final = {
        "learned": {
            "best_epoch": (
                learned_ckpt[
                    "epoch"
                ]
            ),
            "best_val_average_budget_accuracy": (
                learned_ckpt[
                    "best_val_average_budget_accuracy"
                ]
            ),
            "test_average_budget_accuracy": (
                learned_avg
            ),
            "budgets": {},
        },
        "fixed": {
            "best_epoch": (
                fixed_ckpt[
                    "epoch"
                ]
            ),
            "best_val_average_budget_accuracy": (
                fixed_ckpt[
                    "best_val_average_budget_accuracy"
                ]
            ),
            "test_average_budget_accuracy": (
                fixed_avg
            ),
            "budgets": {},
        },
        "random": {
            "best_epoch": (
                random_ckpt[
                    "epoch"
                ]
            ),
            "best_val_average_budget_accuracy": (
                random_ckpt[
                    "best_val_average_budget_accuracy"
                ]
            ),
            "test": random_result,
        },
        "learned_vs_fixed_mcnemar": {},
    }

    for budget in budgets:
        b = str(budget)

        learned = learned_results[b]
        fixed = fixed_results[b]

        print()
        print(
            f"B={budget}"
        )

        print(
            f"  Learned: "
            f"{learned['accuracy']:.2f}% "
            f"loss={learned['loss']:.4f}"
        )

        print(
            f"  Fixed:   "
            f"{fixed['accuracy']:.2f}% "
            f"loss={fixed['loss']:.4f} "
            f"| delta="
            f"{learned['accuracy'] - fixed['accuracy']:+.2f}pp"
        )

        random_summary = (
            random_result[
                "summary"
            ][b]
        )

        print(
            f"  Random:  "
            f"{random_summary['accuracy_mean']:.2f}"
            f"±{random_summary['accuracy_std']:.2f}% "
            f"({args.random_test_repeats} repeats)"
        )

        final["learned"][
            "budgets"
        ][b] = strip_tensor_fields(
            learned
        )

        final["fixed"][
            "budgets"
        ][b] = strip_tensor_fields(
            fixed
        )

        if not torch.equal(
            learned["targets"],
            fixed["targets"],
        ):
            raise RuntimeError(
                f"B={budget}: target order mismatch."
            )

        pair = paired_correctness(
            learned[
                "correctness"
            ],
            fixed[
                "correctness"
            ],
        )

        test = exact_mcnemar(
            pair[
                "learned_only_correct"
            ],
            pair[
                "fixed_only_correct"
            ],
        )

        print(
            f"  McNemar Learned vs Fixed: "
            f"p={test['exact_two_sided_p']:.8g} "
            f"(L-only={pair['learned_only_correct']}, "
            f"F-only={pair['fixed_only_correct']})"
        )

        final[
            "learned_vs_fixed_mcnemar"
        ][b] = {
            "paired_correctness": (
                pair
            ),
            "test": test,
            "learned_minus_fixed_pp": (
                learned[
                    "accuracy"
                ]
                - fixed[
                    "accuracy"
                ]
            ),
        }

    print()
    print(
        f"Average budget accuracy | "
        f"Learned={learned_avg:.2f}% | "
        f"Fixed={fixed_avg:.2f}% | "
        f"Random="
        f"{random_result['average_budget_accuracy_mean']:.2f}"
        f"±{random_result['average_budget_accuracy_std']:.2f}%"
    )

    output_path = (
        root
        / "final_test.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved final test: "
        f"{output_path}"
    )

    # Explicit cleanup before returning.
    del learned_model
    del fixed_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final


# ============================================================
# CLI / main
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
        default=(
            "./outputs/"
            "cifar10_fair_d6_e50_seed42"
        ),
    )

    p.add_argument(
        "--modes",
        type=str,
        default="learned,fixed,random",
        help=(
            "Comma-separated training stages. "
            "Default trains all three."
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        default=6,
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
        "--val-size",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--split-seed",
        type=int,
        default=2026,
    )

    p.add_argument(
        "--loader-seed",
        type=int,
        default=2027,
    )

    p.add_argument(
        "--fixed-order",
        type=str,
        default="0,1,2",
        help=(
            "Canonical fixed full head ordering. "
            "Same ordering is used in every block."
        ),
    )

    p.add_argument(
        "--random-train-seed",
        type=int,
        default=31001,
    )

    p.add_argument(
        "--random-val-seed",
        type=int,
        default=41001,
    )

    p.add_argument(
        "--random-test-seed",
        type=int,
        default=51001,
    )

    p.add_argument(
        "--random-test-repeats",
        type=int,
        default=5,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Model initialization seed. "
            "Run separate experiments with 42/43/44 for 3 seeds."
        ),
    )

    p.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    p.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume each mode from its last.pt if present."
        ),
    )

    p.add_argument(
        "--skip-final-test",
        action="store_true",
        help=(
            "Train requested modes but do not evaluate official test."
        ),
    )

    return p


def validate_args(
    args,
):
    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.depth <= 0:
        raise ValueError(
            "--depth must be > 0."
        )

    if (
        args.embed_dim
        % args.main_heads
        != 0
    ):
        raise ValueError(
            "--embed-dim must be divisible "
            "by --main-heads."
        )

    if (
        args.img_size
        % args.patch_size
        != 0
    ):
        raise ValueError(
            "--img-size must be divisible "
            "by --patch-size."
        )

    if (
        args.tau_start <= 0
        or args.tau_end <= 0
    ):
        raise ValueError(
            "Gumbel temperatures must be positive."
        )

    if args.random_test_repeats <= 0:
        raise ValueError(
            "--random-test-repeats must be > 0."
        )


def main():
    args = build_parser().parse_args()

    validate_args(
        args
    )

    modes = parse_modes(
        args.modes
    )

    device = resolve_device(
        args.device
    )

    fixed_order = (
        parse_fixed_order(
            args.fixed_order,
            args.main_heads,
        )
    )

    budgets = list(
        range(
            args.main_heads + 1
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create split ONCE.
    # CIFAR train set length is discovered from a lightweight dataset object.
    _, eval_transform = (
        build_transforms(
            args.img_size
        )
    )

    split_source = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )

    (
        train_indices,
        val_indices,
    ) = make_split_indices(
        num_samples=len(
            split_source
        ),
        val_size=args.val_size,
        split_seed=args.split_seed,
    )

    (
        train_set,
        val_set,
        test_set,
    ) = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        train_indices=(
            train_indices
        ),
        val_indices=(
            val_indices
        ),
    )

    val_loader = (
        build_eval_loader(
            val_set,
            args.batch_size,
            args.num_workers,
        )
    )

    test_loader = (
        build_eval_loader(
            test_set,
            args.batch_size,
            args.num_workers,
        )
    )

    split_path = (
        output_dir
        / "split_indices.json"
    )

    with split_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "split_seed": (
                    args.split_seed
                ),
                "train_indices": (
                    train_indices
                ),
                "val_indices": (
                    val_indices
                ),
            },
            f,
            indent=2,
        )

    config_path = (
        output_dir
        / "run_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print()
    print(
        "Fair baseline protocol:"
    )
    print(
        f"  train={len(train_set)} | "
        f"val={len(val_set)} | "
        f"test={len(test_set)}"
    )
    print(
        f"  depth={args.depth}, "
        f"epochs={args.epochs}, "
        f"embed={args.embed_dim}, "
        f"heads={args.main_heads}"
    )
    print(
        f"  budgets={budgets}"
    )
    print(
        f"  fixed order="
        f"{fixed_order}"
    )
    print(
        f"  seed={args.seed}"
    )
    print(
        "  TEST IS NOT USED DURING TRAINING."
    )

    best_paths = {}

    for mode in modes:
        best_paths[
            mode
        ] = train_mode(
            mode=mode,
            args=args,
            device=device,
            train_set=train_set,
            val_loader=val_loader,
            fixed_order=fixed_order,
            budgets=budgets,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final test is intentionally blocked unless all three trained checkpoints exist.
    if not args.skip_final_test:
        run_final_test(
            args=args,
            device=device,
            test_loader=test_loader,
            budgets=budgets,
            fixed_order=fixed_order,
        )

    print()
    print("=" * 112)
    print(
        "Fair baseline experiment finished."
    )
    print(
        f"Output: {output_dir}"
    )
    print(
        "For 3-seed reporting, rerun with "
        "--seed 43 and --seed 44 using separate output directories."
    )


if __name__ == "__main__":
    main()