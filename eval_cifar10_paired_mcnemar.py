# eval_cifar10_paired_mcnemar.py
"""
Paired diagnostic evaluation for Mini-to-Main Attention.

재학습 없이 trained best.pt를 로드해 아래만 검사한다.

1) B=1 paired correctness
   Learned vs fixed single head (default: H1)

2) B=2 paired correctness
   Learned vs fixed ordered pair
   (default: H1 direct + H2 mixed)

3) McNemar test
   같은 CIFAR-10 test image 10,000장에 대해
   Learned/Fixed의 correct/incorrect 결과를 paired comparison.

4) B=2 learned direct/mixed role frequency
   block별:
     - direct head frequency
     - mixed head frequency
     - ordered role pair frequency
       예: H1(direct)+H2(mixed)

중요
----
이 평가는 inference-time ablation이다.
fixed model을 별도로 재학습하는 최종 baseline comparison은 아니다.
"""

import argparse
import json
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mini_guided_vit import MiniGuidedViT


# ============================================================
# Utility
# ============================================================

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
        depth=cfg_value(cfg, "depth", 2),
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
        raise KeyError(
            "Checkpoint does not contain key 'model'."
        )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            f"Missing model keys: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected model keys: {unexpected}"
        )

    model.eval()
    return model


def build_test_loader(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = transforms.Compose(
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

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# Fixed allocation override
# ============================================================

def _build_fixed_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    ordered_heads: Sequence[int],
):
    """
    ordered_heads의 앞쪽 head가 direct 우선순위를 갖는다.

    예:
      budget=1, ordered_heads=[1]
        -> H1 direct

      budget=2, ordered_heads=[1,2]
        -> H1 direct, H2 mixed
        (현재 direct_ratio=0.34 / main_heads=3 기준)
    """
    B, H = alloc_logits.shape

    if budget < 0 or budget > H:
        raise ValueError(
            f"Invalid budget={budget}, main_heads={H}"
        )

    if len(ordered_heads) < budget:
        raise ValueError(
            "ordered_heads is shorter than budget."
        )

    selected_heads = list(
        ordered_heads[:budget]
    )

    if len(set(selected_heads)) != budget:
        raise ValueError(
            "Duplicate fixed heads are not allowed."
        )

    for h in selected_heads:
        if h < 0 or h >= H:
            raise ValueError(
                f"Invalid head H{h}; main_heads={H}"
            )

    direct_count = scheduler._get_direct_count(
        budget
    )

    selected = torch.tensor(
        selected_heads,
        dtype=torch.long,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(B, -1)

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

    active_gate = active_mask.to(dtype)
    direct_gate = direct_mask.to(dtype)
    mixed_gate = mixed_mask.to(dtype)
    inactive_gate = inactive_mask.to(dtype)

    rank_scores = torch.zeros_like(
        alloc_logits
    )

    for rank, head in enumerate(
        selected_heads
    ):
        rank_scores[:, head] = float(
            budget - rank
        )

    stats = {
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
def fixed_allocation(
    model: MiniGuidedViT,
    budget: int,
    ordered_heads: Sequence[int],
):
    original_forwards = []

    for block in model.blocks:
        scheduler = block.attn.scheduler

        original_forwards.append(
            (scheduler, scheduler.forward)
        )

        def override_forward(
            self_scheduler,
            alloc_logits,
            budget,
            *,
            _expected_budget=budget,
            _ordered_heads=tuple(ordered_heads),
        ):
            if budget != _expected_budget:
                raise ValueError(
                    f"Expected budget={_expected_budget}, "
                    f"got budget={budget}."
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
        for scheduler, original in (
            original_forwards
        ):
            scheduler.forward = original


# ============================================================
# Paired correctness
# ============================================================

@torch.no_grad()
def collect_correctness(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
):
    """
    동일 test order에 대해 sample별:
      prediction
      target
      correct(bool)
    를 반환한다.
    """
    model.eval()

    predictions = []
    targets_all = []
    correctness = []

    loss_sum = 0.0
    total = 0

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

        predictions.append(
            pred.detach().cpu()
        )
        targets_all.append(
            targets.detach().cpu()
        )
        correctness.append(
            (pred == targets)
            .detach()
            .cpu()
        )

        loss_sum += loss.item()
        total += targets.size(0)

    predictions = torch.cat(
        predictions,
        dim=0,
    )

    targets_all = torch.cat(
        targets_all,
        dim=0,
    )

    correctness = torch.cat(
        correctness,
        dim=0,
    ).bool()

    return {
        "predictions": predictions,
        "targets": targets_all,
        "correctness": correctness,
        "accuracy": (
            100.0
            * correctness.float()
            .mean()
            .item()
        ),
        "loss": loss_sum / total,
        "samples": total,
    }


def paired_correctness_table(
    learned_correct: torch.Tensor,
    fixed_correct: torch.Tensor,
):
    if learned_correct.shape != fixed_correct.shape:
        raise ValueError(
            "Paired correctness tensors have "
            "different shapes."
        )

    learned_correct = learned_correct.bool()
    fixed_correct = fixed_correct.bool()

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

    total = learned_correct.numel()

    if (
        both_correct
        + both_wrong
        + learned_only
        + fixed_only
        != total
    ):
        raise RuntimeError(
            "Paired table count mismatch."
        )

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "learned_only_correct": learned_only,
        "fixed_only_correct": fixed_only,
        "discordant_total": (
            learned_only + fixed_only
        ),
        "samples": total,
        "accuracy_delta_pp_from_pairs": (
            100.0
            * (learned_only - fixed_only)
            / total
        ),
    }


# ============================================================
# McNemar test
# ============================================================

def _log_binomial_pmf_half(
    n: int,
    k: int,
) -> float:
    """
    log[ C(n,k) * 0.5^n ]
    """
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )


def _logsumexp(values: List[float]) -> float:
    if not values:
        return float("-inf")

    m = max(values)

    if math.isinf(m):
        return m

    return (
        m
        + math.log(
            sum(
                math.exp(v - m)
                for v in values
            )
        )
    )


def mcnemar_test(
    learned_only_correct: int,
    fixed_only_correct: int,
):
    """
    McNemar test.

    b = Learned만 맞음
    c = Fixed만 맞음

    H0:
        두 방법의 discordant outcome probability가 동일하다.

    exact two-sided:
        X ~ Binomial(n=b+c, p=0.5)
        p = 2 * P(X <= min(b,c))

    추가로 continuity-corrected chi-square approximation도 반환한다.
    """
    b = int(learned_only_correct)
    c = int(fixed_only_correct)

    n = b + c

    if n == 0:
        return {
            "discordant_total": 0,
            "exact_two_sided_p": 1.0,
            "chi2_continuity_corrected": 0.0,
            "chi2_approx_p": 1.0,
        }

    k = min(b, c)

    log_terms = [
        _log_binomial_pmf_half(
            n,
            i,
        )
        for i in range(k + 1)
    ]

    log_lower_tail = _logsumexp(
        log_terms
    )

    lower_tail = math.exp(
        log_lower_tail
    )

    exact_p = min(
        1.0,
        2.0 * lower_tail,
    )

    # Continuity-corrected McNemar chi-square
    chi2 = (
        (max(abs(b - c) - 1, 0) ** 2)
        / n
    )

    # chi-square(df=1) survival function:
    # P(Chi^2_1 >= x) = erfc(sqrt(x/2))
    approx_p = math.erfc(
        math.sqrt(chi2 / 2.0)
    )

    return {
        "discordant_total": n,
        "exact_two_sided_p": exact_p,
        "chi2_continuity_corrected": chi2,
        "chi2_approx_p": approx_p,
    }


# ============================================================
# B=2 learned direct / mixed role frequency
# ============================================================

def _single_selected_head(
    mask_row: torch.Tensor,
) -> int:
    indices = torch.nonzero(
        mask_row,
        as_tuple=False,
    ).flatten().tolist()

    if len(indices) != 1:
        raise RuntimeError(
            "Expected exactly one selected head "
            f"for role mask, got {indices}."
        )

    return int(indices[0])


@torch.no_grad()
def analyze_b2_role_frequency(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
):
    """
    B=2 learned deterministic allocation에서 block별로:

      direct head frequency
      mixed head frequency
      ordered role pair frequency

    를 계산한다.

    현재 direct_ratio=0.34 / main_heads=3이면
    B=2에서 direct 1개 + mixed 1개가 기대된다.
    """
    model.eval()

    depth = len(model.blocks)
    H = model.main_heads

    direct_counts = [
        torch.zeros(
            H,
            dtype=torch.long,
        )
        for _ in range(depth)
    ]

    mixed_counts = [
        torch.zeros(
            H,
            dtype=torch.long,
        )
        for _ in range(depth)
    ]

    ordered_pair_counts = [
        Counter()
        for _ in range(depth)
    ]

    total_samples = 0

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        _, info_list = model(
            images,
            budget=2,
            return_info=True,
        )

        if len(info_list) != depth:
            raise RuntimeError(
                f"Expected {depth} block infos, "
                f"got {len(info_list)}."
            )

        batch_size = (
            info_list[0][
                "direct_mask"
            ].shape[0]
        )

        for block_idx, info in enumerate(
            info_list
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

            direct_per_sample = (
                direct.long()
                .sum(dim=1)
            )

            mixed_per_sample = (
                mixed.long()
                .sum(dim=1)
            )

            if not torch.all(
                direct_per_sample == 1
            ):
                raise RuntimeError(
                    f"Block {block_idx}: "
                    "B=2 does not have exactly "
                    "one direct head per sample."
                )

            if not torch.all(
                mixed_per_sample == 1
            ):
                raise RuntimeError(
                    f"Block {block_idx}: "
                    "B=2 does not have exactly "
                    "one mixed head per sample."
                )

            direct_counts[
                block_idx
            ] += direct.long().sum(dim=0)

            mixed_counts[
                block_idx
            ] += mixed.long().sum(dim=0)

            for sample_idx in range(
                batch_size
            ):
                direct_head = (
                    _single_selected_head(
                        direct[sample_idx]
                    )
                )

                mixed_head = (
                    _single_selected_head(
                        mixed[sample_idx]
                    )
                )

                label = (
                    f"H{direct_head}(direct)"
                    f"+H{mixed_head}(mixed)"
                )

                ordered_pair_counts[
                    block_idx
                ][label] += 1

        total_samples += batch_size

    blocks = []

    for block_idx in range(depth):
        direct_freq = [
            100.0 * int(v) / total_samples
            for v in direct_counts[
                block_idx
            ]
        ]

        mixed_freq = [
            100.0 * int(v) / total_samples
            for v in mixed_counts[
                block_idx
            ]
        ]

        pair_freq = []

        for label, count in (
            ordered_pair_counts[
                block_idx
            ].most_common()
        ):
            pair_freq.append(
                {
                    "role_pair": label,
                    "count": count,
                    "pct": (
                        100.0
                        * count
                        / total_samples
                    ),
                }
            )

        blocks.append(
            {
                "block": block_idx,
                "direct_head_freq_pct": (
                    direct_freq
                ),
                "mixed_head_freq_pct": (
                    mixed_freq
                ),
                "ordered_role_pairs": (
                    pair_freq
                ),
            }
        )

    return {
        "budget": 2,
        "samples": total_samples,
        "blocks": blocks,
    }


# ============================================================
# Printing
# ============================================================

def print_paired_result(
    title: str,
    learned_result: Dict,
    fixed_result: Dict,
    fixed_label: str,
):
    table = paired_correctness_table(
        learned_result["correctness"],
        fixed_result["correctness"],
    )

    test = mcnemar_test(
        learned_only_correct=(
            table[
                "learned_only_correct"
            ]
        ),
        fixed_only_correct=(
            table[
                "fixed_only_correct"
            ]
        ),
    )

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)

    print(
        f"Learned      acc="
        f"{learned_result['accuracy']:.2f}% "
        f"loss={learned_result['loss']:.4f}"
    )

    print(
        f"{fixed_label:<12} acc="
        f"{fixed_result['accuracy']:.2f}% "
        f"loss={fixed_result['loss']:.4f}"
    )

    print(
        f"Accuracy delta = "
        f"{learned_result['accuracy'] - fixed_result['accuracy']:+.2f}pp"
    )

    print()
    print("Paired correctness:")
    print(
        f"  both correct        : "
        f"{table['both_correct']}"
    )
    print(
        f"  both wrong          : "
        f"{table['both_wrong']}"
    )
    print(
        f"  Learned only correct: "
        f"{table['learned_only_correct']}"
    )
    print(
        f"  Fixed only correct  : "
        f"{table['fixed_only_correct']}"
    )
    print(
        f"  discordant total    : "
        f"{table['discordant_total']}"
    )

    print()
    print("McNemar:")
    print(
        f"  exact two-sided p   = "
        f"{test['exact_two_sided_p']:.8g}"
    )
    print(
        f"  corrected chi2      = "
        f"{test['chi2_continuity_corrected']:.6f}"
    )
    print(
        f"  chi2 approx p       = "
        f"{test['chi2_approx_p']:.8g}"
    )

    if test[
        "exact_two_sided_p"
    ] < 0.05:
        print(
            "  interpretation      : "
            "p < 0.05"
        )
    else:
        print(
            "  interpretation      : "
            "p >= 0.05"
        )

    return {
        "learned": {
            "accuracy": (
                learned_result[
                    "accuracy"
                ]
            ),
            "loss": (
                learned_result["loss"]
            ),
        },
        "fixed": {
            "label": fixed_label,
            "accuracy": (
                fixed_result["accuracy"]
            ),
            "loss": fixed_result["loss"],
        },
        "paired_correctness": table,
        "mcnemar": test,
    }


def print_b2_role_frequency(
    result: Dict,
):
    print()
    print("=" * 96)
    print(
        "B=2 Learned direct / mixed role frequency"
    )
    print("=" * 96)

    for block in result["blocks"]:
        print()
        print(
            f"Block {block['block']}"
        )

        print(
            "  direct: "
            + " ".join(
                f"H{i}:{v:.2f}%"
                for i, v in enumerate(
                    block[
                        "direct_head_freq_pct"
                    ]
                )
            )
        )

        print(
            "  mixed : "
            + " ".join(
                f"H{i}:{v:.2f}%"
                for i, v in enumerate(
                    block[
                        "mixed_head_freq_pct"
                    ]
                )
            )
        )

        print(
            "  ordered role pairs:"
        )

        for item in (
            block["ordered_role_pairs"]
        ):
            print(
                f"    "
                f"{item['role_pair']:<27} "
                f"{item['pct']:6.2f}% "
                f"({item['count']})"
            )


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
            "cifar10_paired_mcnemar"
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

    # 현재 이전 diagnostic 결과에서 찾은 best fixed defaults.
    p.add_argument(
        "--b1-fixed-head",
        type=int,
        default=1,
        help=(
            "B=1 paired comparison fixed head. "
            "Current best fixed single is H1."
        ),
    )

    p.add_argument(
        "--b2-direct-head",
        type=int,
        default=1,
        help=(
            "B=2 fixed direct head. "
            "Current best pair/role uses H1 direct."
        ),
    )

    p.add_argument(
        "--b2-mixed-head",
        type=int,
        default=2,
        help=(
            "B=2 fixed mixed head. "
            "Current best pair/role uses H2 mixed."
        ),
    )

    return p


def main():
    args = build_parser().parse_args()

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

    if model.main_heads != 3:
        raise ValueError(
            "This diagnostic currently expects "
            f"main_heads=3, got {model.main_heads}."
        )

    if (
        args.b2_direct_head
        == args.b2_mixed_head
    ):
        raise ValueError(
            "B=2 direct and mixed heads "
            "must be different."
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

    loader = build_test_loader(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        "Test samples:",
        len(loader.dataset),
    )
    print(
        "Model:",
        f"depth={len(model.blocks)}, "
        f"main_heads={model.main_heads}, "
        f"img_size={img_size}",
    )

    # --------------------------------------------------------
    # B=1 Learned
    # --------------------------------------------------------
    learned_b1 = collect_correctness(
        model=model,
        loader=loader,
        device=device,
        budget=1,
    )

    # B=1 Fixed best head
    with fixed_allocation(
        model=model,
        budget=1,
        ordered_heads=[
            args.b1_fixed_head
        ],
    ):
        fixed_b1 = collect_correctness(
            model=model,
            loader=loader,
            device=device,
            budget=1,
        )

    if not torch.equal(
        learned_b1["targets"],
        fixed_b1["targets"],
    ):
        raise RuntimeError(
            "B=1 paired target order mismatch."
        )

    # --------------------------------------------------------
    # B=2 Learned
    # --------------------------------------------------------
    learned_b2 = collect_correctness(
        model=model,
        loader=loader,
        device=device,
        budget=2,
    )

    # B=2 Fixed best ordered role
    with fixed_allocation(
        model=model,
        budget=2,
        ordered_heads=[
            args.b2_direct_head,
            args.b2_mixed_head,
        ],
    ):
        fixed_b2 = collect_correctness(
            model=model,
            loader=loader,
            device=device,
            budget=2,
        )

    if not torch.equal(
        learned_b2["targets"],
        fixed_b2["targets"],
    ):
        raise RuntimeError(
            "B=2 paired target order mismatch."
        )

    # --------------------------------------------------------
    # Paired + McNemar
    # --------------------------------------------------------
    b1_label = (
        f"Fixed H{args.b1_fixed_head}"
    )

    b2_label = (
        f"Fixed H{args.b2_direct_head}D"
        f"+H{args.b2_mixed_head}M"
    )

    b1_result = print_paired_result(
        title=(
            "B=1 Paired correctness + McNemar"
        ),
        learned_result=learned_b1,
        fixed_result=fixed_b1,
        fixed_label=b1_label,
    )

    b2_result = print_paired_result(
        title=(
            "B=2 Paired correctness + McNemar"
        ),
        learned_result=learned_b2,
        fixed_result=fixed_b2,
        fixed_label=b2_label,
    )

    # --------------------------------------------------------
    # B=2 learned role frequency
    # --------------------------------------------------------
    role_result = (
        analyze_b2_role_frequency(
            model=model,
            loader=loader,
            device=device,
        )
    )

    print_b2_role_frequency(
        role_result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_result = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": (
            checkpoint.get(
                "epoch",
                None,
            )
        ),
        "B1_paired": b1_result,
        "B2_paired": b2_result,
        "B2_learned_role_frequency": (
            role_result
        ),
        "fixed_config": {
            "B1_fixed_head": (
                args.b1_fixed_head
            ),
            "B2_direct_head": (
                args.b2_direct_head
            ),
            "B2_mixed_head": (
                args.b2_mixed_head
            ),
        },
    }

    output_path = (
        output_dir
        / "paired_mcnemar_roles.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 96)
    print(
        "Paired diagnostic evaluation finished."
    )
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()