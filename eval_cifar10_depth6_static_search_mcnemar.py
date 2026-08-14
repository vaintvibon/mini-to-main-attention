# eval_cifar10_depth6_static_search_mcnemar.py
"""
Depth=6 Mini-to-Main diagnostic:
validation-selected block-wise static policy vs learned dynamic routing.

No retraining.

Protocol
--------
CIFAR-10 standard test set (10,000) is deterministically split into:

    policy-search split : 5,000
    held-out eval split : 5,000

On the policy-search split:

B=1
    each of 6 transformer blocks chooses one fixed head:
        H0 / H1 / H2

B=2
    each block chooses one ordered direct/mixed role pair:
        H0D+H1M
        H1D+H0M
        H0D+H2M
        H2D+H0M
        H1D+H2M
        H2D+H1M

Searching all combinations is expensive:
    B=1: 3^6 = 729
    B=2: 6^6 = 46,656

Therefore this script uses coordinate descent:
    1) initialize from Learned's modal per-block choice on search split
    2) for each block, try every legal static choice
    3) keep the choice with highest search accuracy
       (tie-break by lower loss)
    4) repeat until no change or max sweeps reached

Then freeze the selected static policy and compare on the held-out eval split:

    Learned vs Selected Static

with:
    - accuracy / CE loss
    - paired correctness
    - exact two-sided McNemar
    - continuity-corrected McNemar approximation

IMPORTANT LIMITATION
--------------------
The current best.pt checkpoint was itself selected using the standard
CIFAR-10 test set during the previous training experiment.

Therefore this split removes STATIC-POLICY selection leakage between the
search/eval halves, but it does NOT make the 5,000 eval images a pristine
paper-quality test set with respect to MODEL/CHECKPOINT selection.

For final paper protocol:
    train on train split
    select checkpoint/static policy on validation split
    evaluate exactly once on untouched CIFAR-10 test split
"""

import argparse
import copy
import json
import math
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.mini_guided_vit import MiniGuidedViT


# ============================================================
# Reproducibility / utility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device_arg)


def cfg_value(cfg: Dict, key: str, default):
    value = cfg.get(key, default)
    return default if value is None else value


def build_model_from_checkpoint(
    checkpoint: Dict,
    device: torch.device,
) -> MiniGuidedViT:
    cfg = checkpoint.get("args", {})

    model = MiniGuidedViT(
        img_size=cfg_value(cfg, "img_size", 224),
        patch_size=cfg_value(cfg, "patch_size", 16),
        in_chans=3,
        num_classes=10,
        embed_dim=cfg_value(cfg, "embed_dim", 192),
        depth=cfg_value(cfg, "depth", 6),
        main_heads=cfg_value(cfg, "main_heads", 3),
        mlp_ratio=cfg_value(cfg, "mlp_ratio", 4.0),
        mini_heads=cfg_value(cfg, "mini_heads", 1),
        mini_dim=cfg_value(cfg, "mini_dim", 64),
        pool_ratio=cfg_value(cfg, "pool_ratio", 2),
        direct_ratio=cfg_value(cfg, "direct_ratio", 0.34),
        alpha_direct=cfg_value(cfg, "alpha_direct", 1.0),
        alpha_mixed=cfg_value(cfg, "alpha_mixed", 0.2),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=128,
        gumbel_tau=cfg_value(cfg, "tau_end", 0.5),
        use_gumbel=True,
    ).to(device)

    state = checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint does not contain key 'model'.")

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        raise RuntimeError(f"Missing model keys: {missing}")

    if unexpected:
        raise RuntimeError(f"Unexpected model keys: {unexpected}")

    model.eval()
    return model


def build_split_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    split_seed: int,
    search_size: int,
):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    n = len(dataset)

    if search_size <= 0 or search_size >= n:
        raise ValueError(
            f"--search-size must be in [1, {n-1}], got {search_size}."
        )

    generator = torch.Generator()
    generator.manual_seed(split_seed)

    perm = torch.randperm(
        n,
        generator=generator,
    ).tolist()

    search_indices = perm[:search_size]
    eval_indices = perm[search_size:]

    search_set = Subset(
        dataset,
        search_indices,
    )
    eval_set = Subset(
        dataset,
        eval_indices,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    search_loader = DataLoader(
        search_set,
        **loader_kwargs,
    )

    eval_loader = DataLoader(
        eval_set,
        **loader_kwargs,
    )

    return (
        dataset,
        search_indices,
        eval_indices,
        search_loader,
        eval_loader,
    )


# ============================================================
# Fixed scheduler override
# ============================================================

def _build_fixed_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    ordered_heads: Sequence[int],
):
    batch_size, num_heads = alloc_logits.shape

    if budget < 0 or budget > num_heads:
        raise ValueError(
            f"Invalid budget={budget}, main_heads={num_heads}"
        )

    selected_heads = list(
        ordered_heads[:budget]
    )

    if len(selected_heads) != budget:
        raise ValueError(
            f"Need {budget} selected heads, got {selected_heads}."
        )

    if len(set(selected_heads)) != budget:
        raise ValueError(
            f"Duplicate selected heads: {selected_heads}"
        )

    for head in selected_heads:
        if head < 0 or head >= num_heads:
            raise ValueError(
                f"Invalid head H{head}; main_heads={num_heads}."
            )

    direct_count = scheduler._get_direct_count(
        budget
    )

    selected = torch.tensor(
        selected_heads,
        dtype=torch.long,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(
        batch_size,
        -1,
    )

    active_mask = torch.zeros(
        batch_size,
        num_heads,
        dtype=torch.bool,
        device=alloc_logits.device,
    )
    direct_mask = torch.zeros_like(
        active_mask
    )

    if budget > 0:
        active_mask.scatter_(
            1,
            selected,
            True,
        )

    if direct_count > 0:
        direct_mask.scatter_(
            1,
            selected[:, :direct_count],
            True,
        )

    mixed_mask = (
        active_mask & (~direct_mask)
    )
    inactive_mask = ~active_mask

    dtype = alloc_logits.dtype

    selection_scores = torch.zeros_like(
        alloc_logits
    )

    for rank, head in enumerate(
        selected_heads
    ):
        selection_scores[:, head] = float(
            budget - rank
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
        "selection_scores": selection_scores,
        "gumbel_noise": torch.zeros_like(
            alloc_logits
        ),
        "stats": {
            "budget": torch.tensor(
                float(budget),
                dtype=dtype,
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
        },
    }


@contextmanager
def blockwise_fixed_allocation(
    model: MiniGuidedViT,
    budget: int,
    block_configs: Sequence[Sequence[int]],
):
    if len(block_configs) != len(model.blocks):
        raise ValueError(
            f"Expected {len(model.blocks)} block configs, "
            f"got {len(block_configs)}."
        )

    originals = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        scheduler = block.attn.scheduler
        ordered_heads = tuple(
            block_configs[block_idx]
        )

        originals.append(
            (scheduler, scheduler.forward)
        )

        def override_forward(
            self_scheduler,
            alloc_logits,
            budget,
            *,
            _expected_budget=budget,
            _ordered_heads=ordered_heads,
            _block_idx=block_idx,
        ):
            if budget != _expected_budget:
                raise ValueError(
                    f"Block {_block_idx}: expected budget="
                    f"{_expected_budget}, got {budget}."
                )

            return _build_fixed_schedule(
                scheduler=self_scheduler,
                alloc_logits=alloc_logits,
                budget=budget,
                ordered_heads=_ordered_heads,
            )

        scheduler.forward = MethodType(
            override_forward,
            scheduler,
        )

    try:
        yield
    finally:
        for scheduler, original_forward in originals:
            scheduler.forward = original_forward


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
    collect_correctness: bool = False,
):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    correctness = []
    targets_all = []

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

        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(
            logits,
            targets,
            reduction="sum",
        )

        pred = logits.argmax(dim=-1)
        batch_correct = pred.eq(targets)

        total += targets.size(0)
        correct += int(
            batch_correct.sum().item()
        )
        loss_sum += float(
            loss.item()
        )

        if collect_correctness:
            correctness.append(
                batch_correct.detach().cpu()
            )
            targets_all.append(
                targets.detach().cpu()
            )

    result = {
        "accuracy": (
            100.0 * correct / total
        ),
        "loss": loss_sum / total,
        "samples": total,
    }

    if collect_correctness:
        result["correctness"] = (
            torch.cat(correctness).bool()
        )
        result["targets"] = (
            torch.cat(targets_all)
        )

    return result


def evaluate_static(
    model,
    loader,
    device,
    budget,
    config,
    collect_correctness=False,
):
    with blockwise_fixed_allocation(
        model=model,
        budget=budget,
        block_configs=config,
    ):
        return evaluate(
            model=model,
            loader=loader,
            device=device,
            budget=budget,
            collect_correctness=(
                collect_correctness
            ),
        )


# ============================================================
# Learned modal initialization
# ============================================================

def _single_head(mask_row):
    idx = torch.nonzero(
        mask_row,
        as_tuple=False,
    ).flatten().tolist()

    if len(idx) != 1:
        raise RuntimeError(
            f"Expected one head, got {idx}"
        )

    return int(idx[0])


@torch.no_grad()
def learned_modal_config(
    model,
    loader,
    device,
    budget,
):
    """
    Search split에서 Learned routing을 관찰해
    각 block의 modal static choice를 initial policy로 만든다.

    B=1:
        active/direct head 1개

    B=2:
        ordered (direct, mixed) pair
    """
    model.eval()

    depth = len(model.blocks)

    counters = [
        Counter()
        for _ in range(depth)
    ]

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        _, infos = model(
            images,
            budget=budget,
            return_info=True,
        )

        if len(infos) != depth:
            raise RuntimeError(
                f"Expected {depth} block infos, got {len(infos)}."
            )

        for block_idx, info in enumerate(
            infos
        ):
            direct = (
                info["direct_mask"]
                .detach()
                .cpu()
                .bool()
            )

            mixed = (
                info["mixed_mask"]
                .detach()
                .cpu()
                .bool()
            )

            batch_size = direct.shape[0]

            for sample_idx in range(
                batch_size
            ):
                direct_head = (
                    _single_head(
                        direct[sample_idx]
                    )
                )

                if budget == 1:
                    choice = (
                        direct_head,
                    )
                elif budget == 2:
                    mixed_head = (
                        _single_head(
                            mixed[sample_idx]
                        )
                    )
                    choice = (
                        direct_head,
                        mixed_head,
                    )
                else:
                    raise ValueError(
                        "Modal init supports only B=1 or B=2."
                    )

                counters[
                    block_idx
                ][choice] += 1

    config = []
    summary = []

    for block_idx, counter in enumerate(
        counters
    ):
        ranked = counter.most_common()

        if not ranked:
            raise RuntimeError(
                f"No routing observations for block {block_idx}."
            )

        best_choice, best_count = (
            ranked[0]
        )

        total = sum(
            counter.values()
        )

        config.append(
            tuple(best_choice)
        )

        summary.append(
            {
                "block": block_idx,
                "modal_choice": (
                    format_block_choice(
                        budget,
                        best_choice,
                    )
                ),
                "modal_count": (
                    best_count
                ),
                "modal_pct": (
                    100.0
                    * best_count
                    / total
                ),
                "all_choices": [
                    {
                        "choice": (
                            format_block_choice(
                                budget,
                                choice,
                            )
                        ),
                        "count": count,
                        "pct": (
                            100.0
                            * count
                            / total
                        ),
                    }
                    for choice, count
                    in ranked
                ],
            }
        )

    return config, summary


# ============================================================
# Coordinate search
# ============================================================

B1_CHOICES = [
    (0,),
    (1,),
    (2,),
]

B2_CHOICES = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
]


def candidate_choices(
    budget,
):
    if budget == 1:
        return B1_CHOICES

    if budget == 2:
        return B2_CHOICES

    raise ValueError(
        "Only B=1 and B=2 are supported."
    )


def better_result(
    candidate,
    incumbent,
):
    """
    Primary criterion: higher accuracy.
    Tie-break: lower CE loss.
    """
    eps = 1e-12

    if (
        candidate["accuracy"]
        > incumbent["accuracy"] + eps
    ):
        return True

    if (
        abs(
            candidate["accuracy"]
            - incumbent["accuracy"]
        )
        <= eps
        and candidate["loss"]
        < incumbent["loss"] - eps
    ):
        return True

    return False


def coordinate_search(
    model,
    loader,
    device,
    budget,
    initial_config,
    max_sweeps,
):
    config = [
        tuple(x)
        for x in initial_config
    ]

    current_result = evaluate_static(
        model=model,
        loader=loader,
        device=device,
        budget=budget,
        config=config,
        collect_correctness=False,
    )

    history = [
        {
            "stage": "initial",
            "config": (
                format_config(
                    budget,
                    config,
                )
            ),
            "accuracy": (
                current_result[
                    "accuracy"
                ]
            ),
            "loss": (
                current_result[
                    "loss"
                ]
            ),
        }
    ]

    print()
    print(
        f"[B={budget}] Initial static config:"
    )
    print(
        "  ",
        format_config(
            budget,
            config,
        ),
    )
    print(
        f"  search acc="
        f"{current_result['accuracy']:.2f}% "
        f"loss={current_result['loss']:.4f}"
    )

    total_evals = 1

    for sweep in range(
        1,
        max_sweeps + 1,
    ):
        changed = False

        print()
        print(
            f"[B={budget}] Coordinate sweep "
            f"{sweep}/{max_sweeps}"
        )

        for block_idx in range(
            len(config)
        ):
            original_choice = (
                config[block_idx]
            )

            best_choice = (
                original_choice
            )
            best_result = (
                current_result
            )

            print(
                f"  Block {block_idx}:"
            )

            for choice in candidate_choices(
                budget
            ):
                trial_config = list(
                    config
                )
                trial_config[
                    block_idx
                ] = tuple(choice)

                trial_result = (
                    evaluate_static(
                        model=model,
                        loader=loader,
                        device=device,
                        budget=budget,
                        config=trial_config,
                        collect_correctness=False,
                    )
                )

                total_evals += 1

                marker = " "

                if better_result(
                    trial_result,
                    best_result,
                ):
                    best_choice = (
                        tuple(choice)
                    )
                    best_result = (
                        trial_result
                    )
                    marker = "*"

                print(
                    f"    {marker} "
                    f"{format_block_choice(budget, choice):<12} "
                    f"acc={trial_result['accuracy']:.2f}% "
                    f"loss={trial_result['loss']:.4f}"
                )

            if best_choice != original_choice:
                changed = True

            config[
                block_idx
            ] = best_choice

            # Important:
            # best_result corresponds to the same config except
            # this block now uses best_choice.
            current_result = best_result

            print(
                f"    -> keep "
                f"{format_block_choice(budget, best_choice)} "
                f"(acc={current_result['accuracy']:.2f}%)"
            )

        history.append(
            {
                "stage": (
                    f"sweep_{sweep}"
                ),
                "config": (
                    format_config(
                        budget,
                        config,
                    )
                ),
                "accuracy": (
                    current_result[
                        "accuracy"
                    ]
                ),
                "loss": (
                    current_result[
                        "loss"
                    ]
                ),
                "changed": changed,
            }
        )

        print(
            f"[B={budget}] End sweep {sweep}: "
            f"acc={current_result['accuracy']:.2f}% "
            f"loss={current_result['loss']:.4f}"
        )
        print(
            "  ",
            format_config(
                budget,
                config,
            ),
        )

        if not changed:
            print(
                f"[B={budget}] Converged: "
                "no block changed."
            )
            break

    return {
        "config": [
            tuple(x)
            for x in config
        ],
        "search_result": (
            current_result
        ),
        "history": history,
        "total_static_evaluations": (
            total_evals
        ),
    }


# ============================================================
# McNemar
# ============================================================

def paired_table(
    learned_correct,
    static_correct,
):
    learned_correct = (
        learned_correct.bool()
    )
    static_correct = (
        static_correct.bool()
    )

    if (
        learned_correct.shape
        != static_correct.shape
    ):
        raise ValueError(
            "Correctness tensor shape mismatch."
        )

    both_correct = int(
        (
            learned_correct
            & static_correct
        ).sum().item()
    )

    both_wrong = int(
        (
            (~learned_correct)
            & (~static_correct)
        ).sum().item()
    )

    learned_only = int(
        (
            learned_correct
            & (~static_correct)
        ).sum().item()
    )

    static_only = int(
        (
            (~learned_correct)
            & static_correct
        ).sum().item()
    )

    total = (
        learned_correct.numel()
    )

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "learned_only_correct": (
            learned_only
        ),
        "static_only_correct": (
            static_only
        ),
        "discordant_total": (
            learned_only
            + static_only
        ),
        "samples": total,
        "accuracy_delta_pp_from_pairs": (
            100.0
            * (
                learned_only
                - static_only
            )
            / total
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


def mcnemar_exact(
    learned_only,
    static_only,
):
    b = int(learned_only)
    c = int(static_only)

    n = b + c

    if n == 0:
        return {
            "discordant_total": 0,
            "exact_two_sided_p": 1.0,
            "chi2_continuity_corrected": 0.0,
            "chi2_approx_p": 1.0,
        }

    k = min(b, c)

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
        * math.exp(log_lower),
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
        "discordant_total": n,
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
# Formatting / reporting
# ============================================================

def format_block_choice(
    budget,
    choice,
):
    if budget == 1:
        return f"H{choice[0]}"

    if budget == 2:
        return (
            f"H{choice[0]}D"
            f"+H{choice[1]}M"
        )

    return str(choice)


def format_config(
    budget,
    config,
):
    return " | ".join(
        (
            f"Block{i}="
            f"{format_block_choice(budget, choice)}"
        )
        for i, choice
        in enumerate(config)
    )


def serializable_modal(
    modal_summary,
):
    return modal_summary


def report_final(
    budget,
    selected_config,
    search_learned,
    search_static,
    eval_learned,
    eval_static,
):
    if not torch.equal(
        eval_learned["targets"],
        eval_static["targets"],
    ):
        raise RuntimeError(
            f"B={budget}: eval target order mismatch."
        )

    pair = paired_table(
        eval_learned[
            "correctness"
        ],
        eval_static[
            "correctness"
        ],
    )

    test = mcnemar_exact(
        pair[
            "learned_only_correct"
        ],
        pair[
            "static_only_correct"
        ],
    )

    print()
    print("=" * 116)
    print(
        f"B={budget}: Final held-out comparison"
    )
    print("=" * 116)

    print(
        "Selected static policy:"
    )
    print(
        "  ",
        format_config(
            budget,
            selected_config,
        ),
    )

    print()
    print("Policy-search split:")
    print(
        f"  Learned  acc={search_learned['accuracy']:.2f}% "
        f"loss={search_learned['loss']:.4f}"
    )
    print(
        f"  Static   acc={search_static['accuracy']:.2f}% "
        f"loss={search_static['loss']:.4f}"
    )
    print(
        f"  delta    ="
        f"{search_learned['accuracy'] - search_static['accuracy']:+.2f}pp"
    )

    print()
    print("Held-out eval split:")
    print(
        f"  Learned  acc={eval_learned['accuracy']:.2f}% "
        f"loss={eval_learned['loss']:.4f}"
    )
    print(
        f"  Static   acc={eval_static['accuracy']:.2f}% "
        f"loss={eval_static['loss']:.4f}"
    )
    print(
        f"  delta    ="
        f"{eval_learned['accuracy'] - eval_static['accuracy']:+.2f}pp"
    )

    print()
    print("Paired correctness:")
    print(
        f"  both correct         : "
        f"{pair['both_correct']}"
    )
    print(
        f"  both wrong           : "
        f"{pair['both_wrong']}"
    )
    print(
        f"  Learned only correct : "
        f"{pair['learned_only_correct']}"
    )
    print(
        f"  Static only correct  : "
        f"{pair['static_only_correct']}"
    )
    print(
        f"  discordant total     : "
        f"{pair['discordant_total']}"
    )

    print()
    print("McNemar:")
    print(
        f"  exact two-sided p    = "
        f"{test['exact_two_sided_p']:.8g}"
    )
    print(
        f"  corrected chi2       = "
        f"{test['chi2_continuity_corrected']:.6f}"
    )
    print(
        f"  chi2 approx p        = "
        f"{test['chi2_approx_p']:.8g}"
    )

    delta = (
        eval_learned["accuracy"]
        - eval_static["accuracy"]
    )

    if (
        delta > 0
        and test[
            "exact_two_sided_p"
        ] < 0.05
    ):
        interpretation = (
            "Learned is significantly better than "
            "the search-selected block-wise static policy "
            "on this held-out half."
        )
    elif (
        delta < 0
        and test[
            "exact_two_sided_p"
        ] < 0.05
    ):
        interpretation = (
            "The search-selected block-wise static policy "
            "is significantly better than Learned "
            "on this held-out half."
        )
    else:
        interpretation = (
            "No statistically significant paired difference "
            "on this held-out half."
        )

    print(
        f"  interpretation       : "
        f"{interpretation}"
    )

    return {
        "selected_static_config": (
            format_config(
                budget,
                selected_config,
            )
        ),
        "search_split": {
            "learned": {
                "accuracy": (
                    search_learned[
                        "accuracy"
                    ]
                ),
                "loss": (
                    search_learned[
                        "loss"
                    ]
                ),
            },
            "static": {
                "accuracy": (
                    search_static[
                        "accuracy"
                    ]
                ),
                "loss": (
                    search_static[
                        "loss"
                    ]
                ),
            },
            "learned_minus_static_pp": (
                search_learned[
                    "accuracy"
                ]
                - search_static[
                    "accuracy"
                ]
            ),
        },
        "heldout_eval": {
            "learned": {
                "accuracy": (
                    eval_learned[
                        "accuracy"
                    ]
                ),
                "loss": (
                    eval_learned[
                        "loss"
                    ]
                ),
            },
            "static": {
                "accuracy": (
                    eval_static[
                        "accuracy"
                    ]
                ),
                "loss": (
                    eval_static[
                        "loss"
                    ]
                ),
            },
            "learned_minus_static_pp": (
                delta
            ),
        },
        "paired_correctness": pair,
        "mcnemar": test,
        "interpretation": interpretation,
    }


# ============================================================
# Optional random restart
# ============================================================

def random_initial_config(
    budget,
    depth,
    rng,
):
    choices = candidate_choices(
        budget
    )

    return [
        tuple(
            rng.choice(choices)
        )
        for _ in range(depth)
    ]


def search_with_optional_restarts(
    model,
    loader,
    device,
    budget,
    modal_config,
    max_sweeps,
    random_restarts,
    seed,
):
    runs = []

    print()
    print("#" * 116)
    print(
        f"B={budget}: modal-initialized coordinate search"
    )
    print("#" * 116)

    modal_run = coordinate_search(
        model=model,
        loader=loader,
        device=device,
        budget=budget,
        initial_config=modal_config,
        max_sweeps=max_sweeps,
    )

    modal_run[
        "initialization"
    ] = "learned_modal"

    runs.append(
        modal_run
    )

    rng = random.Random(
        seed + 1000 * budget
    )

    for restart_idx in range(
        random_restarts
    ):
        initial = random_initial_config(
            budget=budget,
            depth=len(
                modal_config
            ),
            rng=rng,
        )

        print()
        print("#" * 116)
        print(
            f"B={budget}: random restart "
            f"{restart_idx + 1}/{random_restarts}"
        )
        print("#" * 116)

        run = coordinate_search(
            model=model,
            loader=loader,
            device=device,
            budget=budget,
            initial_config=initial,
            max_sweeps=max_sweeps,
        )

        run[
            "initialization"
        ] = (
            f"random_{restart_idx + 1}"
        )

        runs.append(
            run
        )

    best_run = runs[0]

    for run in runs[1:]:
        if better_result(
            run["search_result"],
            best_run["search_result"],
        ):
            best_run = run

    return best_run, runs


# ============================================================
# Main
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

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
            "cifar10_depth6_static_search"
        ),
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
        "--device",
        type=str,
        default="auto",
    )

    p.add_argument(
        "--split-seed",
        type=int,
        default=2026,
    )

    p.add_argument(
        "--search-size",
        type=int,
        default=5000,
        help=(
            "Number of CIFAR-10 test images used "
            "only for static-policy search."
        ),
    )

    p.add_argument(
        "--max-sweeps",
        type=int,
        default=3,
        help=(
            "Maximum coordinate-descent sweeps."
        ),
    )

    p.add_argument(
        "--random-restarts",
        type=int,
        default=0,
        help=(
            "Optional extra random coordinate-search restarts. "
            "0 is the default for runtime efficiency."
        ),
    )

    return p


def main():
    args = build_parser().parse_args()

    set_seed(
        args.split_seed
    )

    device = resolve_device(
        args.device
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model_from_checkpoint(
        checkpoint,
        device,
    )

    depth = len(
        model.blocks
    )

    if depth != 6:
        raise ValueError(
            "This script targets the current depth=6 experiment, "
            f"but checkpoint depth={depth}."
        )

    if model.main_heads != 3:
        raise ValueError(
            "This script expects main_heads=3, "
            f"got {model.main_heads}."
        )

    cfg = checkpoint.get(
        "args",
        {},
    )

    img_size = cfg_value(
        cfg,
        "img_size",
        224,
    )

    (
        full_dataset,
        search_indices,
        eval_indices,
        search_loader,
        eval_loader,
    ) = build_split_loaders(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        search_size=args.search_size,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Full CIFAR-10 test:",
        len(full_dataset),
    )
    print(
        "Policy-search split:",
        len(search_indices),
    )
    print(
        "Held-out eval split:",
        len(eval_indices),
    )
    print(
        "Split seed:",
        args.split_seed,
    )
    print(
        "Model:",
        f"depth={depth}, "
        f"main_heads={model.main_heads}, "
        f"img_size={img_size}",
    )
    print(
        "Coordinate search:",
        f"max_sweeps={args.max_sweeps}, "
        f"random_restarts={args.random_restarts}",
    )

    final_results = {}

    for budget in [1, 2]:
        print()
        print("=" * 116)
        print(
            f"B={budget}: Learn modal initialization "
            "on policy-search split"
        )
        print("=" * 116)

        modal_config, modal_summary = (
            learned_modal_config(
                model=model,
                loader=search_loader,
                device=device,
                budget=budget,
            )
        )

        print(
            "Modal initial config:"
        )
        print(
            "  ",
            format_config(
                budget,
                modal_config,
            ),
        )

        for item in modal_summary:
            print(
                f"  Block {item['block']}: "
                f"{item['modal_choice']} "
                f"{item['modal_pct']:.2f}%"
            )

        search_learned = evaluate(
            model=model,
            loader=search_loader,
            device=device,
            budget=budget,
            collect_correctness=False,
        )

        best_search, all_search_runs = (
            search_with_optional_restarts(
                model=model,
                loader=search_loader,
                device=device,
                budget=budget,
                modal_config=modal_config,
                max_sweeps=args.max_sweeps,
                random_restarts=(
                    args.random_restarts
                ),
                seed=args.split_seed,
            )
        )

        selected_config = (
            best_search[
                "config"
            ]
        )

        search_static = (
            best_search[
                "search_result"
            ]
        )

        print()
        print(
            f"B={budget} selected policy:"
        )
        print(
            "  ",
            format_config(
                budget,
                selected_config,
            ),
        )
        print(
            f"  search static acc="
            f"{search_static['accuracy']:.2f}% "
            f"loss={search_static['loss']:.4f}"
        )
        print(
            f"  search learned acc="
            f"{search_learned['accuracy']:.2f}% "
            f"loss={search_learned['loss']:.4f}"
        )

        # Final held-out comparison.
        eval_learned = evaluate(
            model=model,
            loader=eval_loader,
            device=device,
            budget=budget,
            collect_correctness=True,
        )

        eval_static = evaluate_static(
            model=model,
            loader=eval_loader,
            device=device,
            budget=budget,
            config=selected_config,
            collect_correctness=True,
        )

        final_report = report_final(
            budget=budget,
            selected_config=(
                selected_config
            ),
            search_learned=(
                search_learned
            ),
            search_static=(
                search_static
            ),
            eval_learned=(
                eval_learned
            ),
            eval_static=(
                eval_static
            ),
        )

        final_results[
            f"B{budget}"
        ] = {
            "modal_initialization": (
                serializable_modal(
                    modal_summary
                )
            ),
            "selected_search_run": {
                "initialization": (
                    best_search[
                        "initialization"
                    ]
                ),
                "config": (
                    format_config(
                        budget,
                        selected_config,
                    )
                ),
                "search_accuracy": (
                    search_static[
                        "accuracy"
                    ]
                ),
                "search_loss": (
                    search_static[
                        "loss"
                    ]
                ),
                "total_static_evaluations": (
                    best_search[
                        "total_static_evaluations"
                    ]
                ),
                "history": (
                    best_search[
                        "history"
                    ]
                ),
            },
            "all_search_runs": [
                {
                    "initialization": (
                        run[
                            "initialization"
                        ]
                    ),
                    "config": (
                        format_config(
                            budget,
                            run[
                                "config"
                            ],
                        )
                    ),
                    "search_accuracy": (
                        run[
                            "search_result"
                        ][
                            "accuracy"
                        ]
                    ),
                    "search_loss": (
                        run[
                            "search_result"
                        ][
                            "loss"
                        ]
                    ),
                    "total_static_evaluations": (
                        run[
                            "total_static_evaluations"
                        ]
                    ),
                }
                for run
                in all_search_runs
            ],
            "final_comparison": (
                final_report
            ),
        }

    output_path = (
        output_dir
        / "depth6_static_search_mcnemar.json"
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
                "search_indices": (
                    search_indices
                ),
                "eval_indices": (
                    eval_indices
                ),
            },
            f,
            indent=2,
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "warning": (
                    "This is a stronger exploratory diagnostic, "
                    "not a pristine final test protocol, because "
                    "the checkpoint was previously selected using "
                    "the full CIFAR-10 test set."
                ),
                "checkpoint": str(
                    checkpoint_path
                ),
                "checkpoint_epoch": (
                    checkpoint.get(
                        "epoch",
                        None,
                    )
                ),
                "split_seed": (
                    args.split_seed
                ),
                "search_size": (
                    len(search_indices)
                ),
                "heldout_eval_size": (
                    len(eval_indices)
                ),
                "max_sweeps": (
                    args.max_sweeps
                ),
                "random_restarts": (
                    args.random_restarts
                ),
                "results": (
                    final_results
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 116)
    print(
        "Depth=6 static-search diagnostic finished."
    )
    print(
        "Results:",
        output_path,
    )
    print(
        "Split indices:",
        split_path,
    )
    print()
    print(
        "IMPORTANT: checkpoint selection previously touched "
        "the standard test set. Treat this result as exploratory."
    )


if __name__ == "__main__":
    main()