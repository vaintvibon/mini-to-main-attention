# eval_cifar10_exhaustive_blockwise_fixed.py
"""
Exhaustive block-wise fixed-policy evaluation for the current depth=2,
main_heads=3 CIFAR-10 sanity checkpoint.

No retraining.

B=1:
    Each block chooses exactly one fixed head from {H0,H1,H2}.
    depth=2 => 3^2 = 9 policies.

B=2:
    Each block chooses one ordered direct/mixed pair from:
        H0D+H1M
        H1D+H0M
        H0D+H2M
        H2D+H0M
        H1D+H2M
        H2D+H1M
    depth=2 => 6^2 = 36 policies.

For each budget:
    1) Evaluate Learned.
    2) Exhaustively evaluate all static block-wise policies.
    3) Rank policies by accuracy, then loss.
    4) Compare Learned vs best static policy with paired correctness
       and exact two-sided McNemar test.

IMPORTANT:
    The "best static" here is selected using the same test set.
    Therefore the best-policy comparison is exploratory and slightly
    optimistic for the static baseline. For a final paper-quality result,
    choose the static policy on a validation set and evaluate it once on test.
"""

import argparse
import itertools
import json
import math
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


def build_test_loader(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
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

    selected_heads = list(ordered_heads[:budget])

    if len(selected_heads) != budget:
        raise ValueError(
            f"Need {budget} heads, got {selected_heads}."
        )

    if len(set(selected_heads)) != budget:
        raise ValueError("Duplicate selected heads.")

    for head in selected_heads:
        if head < 0 or head >= num_heads:
            raise ValueError(
                f"Invalid head H{head}; main_heads={num_heads}."
            )

    direct_count = scheduler._get_direct_count(budget)

    selected = torch.tensor(
        selected_heads,
        dtype=torch.long,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(batch_size, -1)

    active_mask = torch.zeros(
        batch_size,
        num_heads,
        dtype=torch.bool,
        device=alloc_logits.device,
    )
    direct_mask = torch.zeros_like(active_mask)

    if budget > 0:
        active_mask.scatter_(1, selected, True)

    if direct_count > 0:
        direct_mask.scatter_(
            1,
            selected[:, :direct_count],
            True,
        )

    mixed_mask = active_mask & (~direct_mask)
    inactive_mask = ~active_mask

    dtype = alloc_logits.dtype

    selection_scores = torch.zeros_like(alloc_logits)
    for rank, head in enumerate(selected_heads):
        selection_scores[:, head] = float(budget - rank)

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
        "gumbel_noise": torch.zeros_like(alloc_logits),
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

    for block_idx, block in enumerate(model.blocks):
        scheduler = block.attn.scheduler
        ordered_heads = tuple(block_configs[block_idx])

        originals.append((scheduler, scheduler.forward))

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
def collect_result(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
):
    model.eval()

    correctness = []
    targets_all = []

    total = 0
    loss_sum = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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
        correct = pred.eq(targets)

        correctness.append(correct.detach().cpu())
        targets_all.append(targets.detach().cpu())

        total += targets.size(0)
        loss_sum += loss.item()

    correctness = torch.cat(correctness).bool()
    targets_all = torch.cat(targets_all)

    return {
        "accuracy": (
            100.0
            * correctness.float().mean().item()
        ),
        "loss": loss_sum / total,
        "samples": total,
        "correctness": correctness,
        "targets": targets_all,
    }


# ============================================================
# McNemar
# ============================================================

def paired_table(
    learned_correct: torch.Tensor,
    fixed_correct: torch.Tensor,
):
    learned_correct = learned_correct.bool()
    fixed_correct = fixed_correct.bool()

    if learned_correct.shape != fixed_correct.shape:
        raise ValueError(
            "Correctness tensors have different shapes."
        )

    both_correct = int(
        (learned_correct & fixed_correct).sum().item()
    )
    both_wrong = int(
        ((~learned_correct) & (~fixed_correct)).sum().item()
    )
    learned_only = int(
        (learned_correct & (~fixed_correct)).sum().item()
    )
    fixed_only = int(
        ((~learned_correct) & fixed_correct).sum().item()
    )

    total = learned_correct.numel()

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "learned_only_correct": learned_only,
        "fixed_only_correct": fixed_only,
        "discordant_total": learned_only + fixed_only,
        "samples": total,
        "accuracy_delta_pp_from_pairs": (
            100.0 * (learned_only - fixed_only) / total
        ),
    }


def _log_binomial_pmf_half(n: int, k: int) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )


def _logsumexp(values: List[float]) -> float:
    m = max(values)
    return m + math.log(
        sum(math.exp(v - m) for v in values)
    )


def mcnemar_exact(
    learned_only_correct: int,
    fixed_only_correct: int,
):
    b = int(learned_only_correct)
    c = int(fixed_only_correct)
    n = b + c

    if n == 0:
        return {
            "exact_two_sided_p": 1.0,
            "discordant_total": 0,
            "chi2_continuity_corrected": 0.0,
            "chi2_approx_p": 1.0,
        }

    k = min(b, c)

    log_lower = _logsumexp(
        [
            _log_binomial_pmf_half(n, i)
            for i in range(k + 1)
        ]
    )

    exact_p = min(
        1.0,
        2.0 * math.exp(log_lower),
    )

    chi2 = (
        max(abs(b - c) - 1, 0) ** 2
    ) / n

    approx_p = math.erfc(
        math.sqrt(chi2 / 2.0)
    )

    return {
        "exact_two_sided_p": exact_p,
        "discordant_total": n,
        "chi2_continuity_corrected": chi2,
        "chi2_approx_p": approx_p,
    }


# ============================================================
# Config generation / formatting
# ============================================================

def b1_configs() -> List[Tuple[Tuple[int], Tuple[int]]]:
    choices = [(0,), (1,), (2,)]
    return list(itertools.product(choices, repeat=2))


def b2_role_choices() -> List[Tuple[int, int]]:
    return [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ]


def b2_configs():
    choices = b2_role_choices()
    return list(itertools.product(choices, repeat=2))


def format_config(
    budget: int,
    config,
) -> str:
    parts = []

    for block_idx, heads in enumerate(config):
        if budget == 1:
            parts.append(
                f"Block{block_idx}=H{heads[0]}"
            )
        elif budget == 2:
            parts.append(
                f"Block{block_idx}=H{heads[0]}D+H{heads[1]}M"
            )

    return " | ".join(parts)


# ============================================================
# Exhaustive sweep
# ============================================================

def evaluate_exhaustive(
    model,
    loader,
    device,
    budget,
    configs,
):
    results = []

    for idx, config in enumerate(configs, start=1):
        with blockwise_fixed_allocation(
            model=model,
            budget=budget,
            block_configs=config,
        ):
            result = collect_result(
                model=model,
                loader=loader,
                device=device,
                budget=budget,
            )

        item = {
            "config": config,
            "config_str": format_config(
                budget,
                config,
            ),
            "accuracy": result["accuracy"],
            "loss": result["loss"],
            "correctness": result["correctness"],
            "targets": result["targets"],
        }

        results.append(item)

        print(
            f"[B={budget}] "
            f"{idx:02d}/{len(configs):02d} | "
            f"acc={item['accuracy']:.2f}% "
            f"loss={item['loss']:.4f} | "
            f"{item['config_str']}"
        )

    results.sort(
        key=lambda x: (
            -x["accuracy"],
            x["loss"],
        )
    )

    return results


def report_budget(
    budget,
    learned,
    ranked_results,
    top_k,
):
    best = ranked_results[0]

    if not torch.equal(
        learned["targets"],
        best["targets"],
    ):
        raise RuntimeError(
            f"B={budget}: target order mismatch."
        )

    pair = paired_table(
        learned["correctness"],
        best["correctness"],
    )

    mcnemar = mcnemar_exact(
        pair["learned_only_correct"],
        pair["fixed_only_correct"],
    )

    print()
    print("=" * 110)
    print(
        f"B={budget}: Exhaustive block-wise fixed ranking"
    )
    print("=" * 110)

    print(
        f"Learned: acc={learned['accuracy']:.2f}% "
        f"loss={learned['loss']:.4f}"
    )

    print()
    print(f"Top {min(top_k, len(ranked_results))} static policies:")

    for rank, item in enumerate(
        ranked_results[:top_k],
        start=1,
    ):
        delta = (
            learned["accuracy"]
            - item["accuracy"]
        )

        print(
            f"  #{rank:02d} "
            f"acc={item['accuracy']:.2f}% "
            f"loss={item['loss']:.4f} "
            f"learned-static={delta:+.2f}pp | "
            f"{item['config_str']}"
        )

    print()
    print("Best static policy:")
    print(f"  {best['config_str']}")
    print(
        f"  static acc={best['accuracy']:.2f}% "
        f"loss={best['loss']:.4f}"
    )
    print(
        f"  learned acc={learned['accuracy']:.2f}% "
        f"loss={learned['loss']:.4f}"
    )
    print(
        f"  accuracy delta="
        f"{learned['accuracy'] - best['accuracy']:+.2f}pp"
    )

    print()
    print("Paired correctness vs best static:")
    print(
        f"  both correct        : {pair['both_correct']}"
    )
    print(
        f"  both wrong          : {pair['both_wrong']}"
    )
    print(
        f"  Learned only correct: {pair['learned_only_correct']}"
    )
    print(
        f"  Static only correct : {pair['fixed_only_correct']}"
    )
    print(
        f"  discordant total    : {pair['discordant_total']}"
    )

    print()
    print("McNemar vs best static:")
    print(
        f"  exact two-sided p   = "
        f"{mcnemar['exact_two_sided_p']:.8g}"
    )
    print(
        f"  corrected chi2      = "
        f"{mcnemar['chi2_continuity_corrected']:.6f}"
    )
    print(
        f"  chi2 approx p       = "
        f"{mcnemar['chi2_approx_p']:.8g}"
    )

    if (
        learned["accuracy"] > best["accuracy"]
        and mcnemar["exact_two_sided_p"] < 0.05
    ):
        interpretation = (
            "Learned is significantly better than "
            "the best exhaustively searched static policy "
            "on this test set."
        )
    elif (
        learned["accuracy"] < best["accuracy"]
        and mcnemar["exact_two_sided_p"] < 0.05
    ):
        interpretation = (
            "The best exhaustively searched static policy "
            "is significantly better than Learned."
        )
    else:
        interpretation = (
            "No statistically significant paired difference "
            "between Learned and the best searched static policy."
        )

    print(
        f"  interpretation      : {interpretation}"
    )

    serializable_ranking = []

    for rank, item in enumerate(
        ranked_results,
        start=1,
    ):
        serializable_ranking.append(
            {
                "rank": rank,
                "config": item["config_str"],
                "accuracy": item["accuracy"],
                "loss": item["loss"],
                "learned_minus_static_pp": (
                    learned["accuracy"]
                    - item["accuracy"]
                ),
            }
        )

    return {
        "learned": {
            "accuracy": learned["accuracy"],
            "loss": learned["loss"],
        },
        "best_static": {
            "config": best["config_str"],
            "accuracy": best["accuracy"],
            "loss": best["loss"],
        },
        "learned_minus_best_static_pp": (
            learned["accuracy"]
            - best["accuracy"]
        ),
        "paired_correctness": pair,
        "mcnemar": mcnemar,
        "interpretation": interpretation,
        "ranking": serializable_ranking,
    }


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
        default="./outputs/cifar10_exhaustive_blockwise_fixed",
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
        "--top-k",
        type=int,
        default=10,
    )

    return p


def main():
    args = build_parser().parse_args()

    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model_from_checkpoint(
        checkpoint,
        device,
    )

    if len(model.blocks) != 2:
        raise ValueError(
            "Current exhaustive script expects depth=2, "
            f"got depth={len(model.blocks)}."
        )

    if model.main_heads != 3:
        raise ValueError(
            "Current exhaustive script expects main_heads=3, "
            f"got {model.main_heads}."
        )

    cfg = checkpoint.get("args", {})
    img_size = cfg_value(cfg, "img_size", 224)

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
        checkpoint.get("epoch", "unknown"),
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

    print()
    print("Evaluating Learned B=1...")
    learned_b1 = collect_result(
        model=model,
        loader=loader,
        device=device,
        budget=1,
    )

    print("Evaluating Learned B=2...")
    learned_b2 = collect_result(
        model=model,
        loader=loader,
        device=device,
        budget=2,
    )

    configs_b1 = b1_configs()
    configs_b2 = b2_configs()

    print()
    print(
        f"Starting exhaustive B=1 sweep: "
        f"{len(configs_b1)} policies"
    )

    ranked_b1 = evaluate_exhaustive(
        model=model,
        loader=loader,
        device=device,
        budget=1,
        configs=configs_b1,
    )

    print()
    print(
        f"Starting exhaustive B=2 sweep: "
        f"{len(configs_b2)} policies"
    )

    ranked_b2 = evaluate_exhaustive(
        model=model,
        loader=loader,
        device=device,
        budget=2,
        configs=configs_b2,
    )

    b1_summary = report_budget(
        budget=1,
        learned=learned_b1,
        ranked_results=ranked_b1,
        top_k=args.top_k,
    )

    b2_summary = report_budget(
        budget=2,
        learned=learned_b2,
        ranked_results=ranked_b2,
        top_k=args.top_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "exhaustive_blockwise_fixed.json"
    )

    final = {
        "warning": (
            "Best static policy was selected on the same test set. "
            "Treat the comparison as exploratory. For final reporting, "
            "select the policy on validation data and evaluate once on test."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "num_B1_policies": len(configs_b1),
        "num_B2_policies": len(configs_b2),
        "B1": b1_summary,
        "B2": b2_summary,
    }

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

    print()
    print("=" * 110)
    print("Exhaustive evaluation finished.")
    print("Saved:", output_path)
    print()
    print(
        "NOTE: best static was selected on the same test set; "
        "this is exploratory, not final paper protocol."
    )


if __name__ == "__main__":
    main()