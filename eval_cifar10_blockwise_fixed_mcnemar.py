# eval_cifar10_blockwise_fixed_mcnemar.py
"""
Learned dynamic routing vs block-wise fixed routing.

No retraining. Loads the trained best.pt checkpoint and evaluates:

B=1
  Learned
  vs
  Block0 -> H2
  Block1 -> H1

B=2
  Learned
  vs
  Block0 -> H2(direct) + H1(mixed)
  Block1 -> H1(direct) + H2(mixed)

For each budget:
  - accuracy / loss
  - paired correctness table
  - exact two-sided McNemar p-value
  - continuity-corrected chi-square approximation

This isolates the key question:
"Is sample-dependent routing useful beyond simply assigning each block
its preferred static head/role configuration?"
"""

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence

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
# Fixed scheduler construction
# ============================================================

def _build_fixed_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    ordered_heads: Sequence[int],
):
    """
    ordered_heads의 앞쪽 head가 direct 우선순위를 갖는다.

    B=1, [2]
      -> H2 direct

    B=2, [2, 1]
      -> H2 direct + H1 mixed
      (현재 direct_ratio=0.34, main_heads=3 기준)
    """
    batch_size, num_heads = alloc_logits.shape

    if budget < 0 or budget > num_heads:
        raise ValueError(
            f"Invalid budget={budget} for num_heads={num_heads}"
        )

    if len(ordered_heads) < budget:
        raise ValueError("ordered_heads is shorter than budget.")

    selected_heads = list(ordered_heads[:budget])

    if len(set(selected_heads)) != budget:
        raise ValueError("Duplicate fixed heads are not allowed.")

    for head in selected_heads:
        if head < 0 or head >= num_heads:
            raise ValueError(
                f"Invalid head H{head}; num_heads={num_heads}"
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

    active_gate = active_mask.to(dtype)
    direct_gate = direct_mask.to(dtype)
    mixed_gate = mixed_mask.to(dtype)
    inactive_gate = inactive_mask.to(dtype)

    selection_scores = torch.zeros_like(alloc_logits)
    for rank, head in enumerate(selected_heads):
        selection_scores[:, head] = float(budget - rank)

    stats = {
        "budget": torch.tensor(
            float(budget),
            dtype=dtype,
            device=alloc_logits.device,
        ),
        "active_count_mean": active_mask.float().sum(dim=1).mean().detach(),
        "direct_count_mean": direct_mask.float().sum(dim=1).mean().detach(),
        "mixed_count_mean": mixed_mask.float().sum(dim=1).mean().detach(),
        "inactive_count_mean": inactive_mask.float().sum(dim=1).mean().detach(),
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
        "selection_scores": selection_scores,
        "gumbel_noise": torch.zeros_like(alloc_logits),
        "stats": stats,
    }


@contextmanager
def blockwise_fixed_allocation(
    model: MiniGuidedViT,
    budget: int,
    block_configs: Sequence[Sequence[int]],
):
    """
    block마다 서로 다른 fixed allocation을 적용한다.

    Example:
      B=1:
        block_configs = [[2], [1]]

      B=2:
        block_configs = [[2,1], [1,2]]
    """
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
# Inference and correctness
# ============================================================

@torch.no_grad()
def collect_correctness(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
):
    model.eval()

    preds_all = []
    targets_all = []
    correct_all = []

    loss_sum = 0.0
    total = 0

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

        preds_all.append(pred.detach().cpu())
        targets_all.append(targets.detach().cpu())
        correct_all.append(correct.detach().cpu())

        loss_sum += loss.item()
        total += targets.size(0)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    correct_all = torch.cat(correct_all).bool()

    return {
        "predictions": preds_all,
        "targets": targets_all,
        "correctness": correct_all,
        "accuracy": 100.0 * correct_all.float().mean().item(),
        "loss": loss_sum / total,
        "samples": total,
    }


# ============================================================
# Paired correctness + McNemar
# ============================================================

def paired_table(
    learned_correct: torch.Tensor,
    fixed_correct: torch.Tensor,
):
    if learned_correct.shape != fixed_correct.shape:
        raise ValueError("Correctness tensors have different shapes.")

    learned_correct = learned_correct.bool()
    fixed_correct = fixed_correct.bool()

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

    if both_correct + both_wrong + learned_only + fixed_only != total:
        raise RuntimeError("Paired correctness count mismatch.")

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
    if not values:
        return float("-inf")

    m = max(values)

    if math.isinf(m):
        return m

    return m + math.log(
        sum(math.exp(v - m) for v in values)
    )


def mcnemar_test(
    learned_only_correct: int,
    fixed_only_correct: int,
):
    """
    Exact two-sided McNemar:
      b = Learned only correct
      c = Fixed only correct
      n = b + c

    Under H0:
      X ~ Binomial(n, 0.5)

    p = 2 * P(X <= min(b,c))
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
        "discordant_total": n,
        "exact_two_sided_p": exact_p,
        "chi2_continuity_corrected": chi2,
        "chi2_approx_p": approx_p,
    }


# ============================================================
# Reporting
# ============================================================

def config_to_string(
    budget: int,
    block_configs: Sequence[Sequence[int]],
) -> str:
    parts = []

    for block_idx, heads in enumerate(block_configs):
        if budget == 1:
            parts.append(
                f"Block{block_idx}=H{heads[0]}"
            )
        elif budget == 2:
            parts.append(
                f"Block{block_idx}=H{heads[0]}D+H{heads[1]}M"
            )
        else:
            parts.append(
                f"Block{block_idx}={list(heads)}"
            )

    return " | ".join(parts)


def report(
    budget: int,
    learned_result: Dict,
    fixed_result: Dict,
    block_configs: Sequence[Sequence[int]],
):
    if not torch.equal(
        learned_result["targets"],
        fixed_result["targets"],
    ):
        raise RuntimeError(
            f"B={budget}: paired target order mismatch."
        )

    table = paired_table(
        learned_result["correctness"],
        fixed_result["correctness"],
    )

    test = mcnemar_test(
        table["learned_only_correct"],
        table["fixed_only_correct"],
    )

    config_str = config_to_string(
        budget,
        block_configs,
    )

    print()
    print("=" * 100)
    print(
        f"B={budget}: Learned vs Block-wise Fixed"
    )
    print("=" * 100)

    print(
        f"Block-wise fixed config: {config_str}"
    )
    print()

    print(
        f"Learned          acc={learned_result['accuracy']:.2f}% "
        f"loss={learned_result['loss']:.4f}"
    )
    print(
        f"Block-wise Fixed acc={fixed_result['accuracy']:.2f}% "
        f"loss={fixed_result['loss']:.4f}"
    )
    print(
        f"Accuracy delta   = "
        f"{learned_result['accuracy'] - fixed_result['accuracy']:+.2f}pp"
    )

    print()
    print("Paired correctness:")
    print(
        f"  both correct        : {table['both_correct']}"
    )
    print(
        f"  both wrong          : {table['both_wrong']}"
    )
    print(
        f"  Learned only correct: {table['learned_only_correct']}"
    )
    print(
        f"  Fixed only correct  : {table['fixed_only_correct']}"
    )
    print(
        f"  discordant total    : {table['discordant_total']}"
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

    if (
        learned_result["accuracy"]
        > fixed_result["accuracy"]
        and test["exact_two_sided_p"] < 0.05
    ):
        interpretation = (
            "Learned is significantly better than "
            "this block-wise fixed baseline."
        )
    elif (
        learned_result["accuracy"]
        < fixed_result["accuracy"]
        and test["exact_two_sided_p"] < 0.05
    ):
        interpretation = (
            "Block-wise fixed is significantly better "
            "than Learned."
        )
    else:
        interpretation = (
            "No statistically significant paired difference."
        )

    print(
        f"  interpretation      : {interpretation}"
    )

    return {
        "budget": budget,
        "blockwise_fixed_config": config_str,
        "learned": {
            "accuracy": learned_result["accuracy"],
            "loss": learned_result["loss"],
        },
        "blockwise_fixed": {
            "accuracy": fixed_result["accuracy"],
            "loss": fixed_result["loss"],
        },
        "accuracy_delta_pp": (
            learned_result["accuracy"]
            - fixed_result["accuracy"]
        ),
        "paired_correctness": table,
        "mcnemar": test,
        "interpretation": interpretation,
    }


# ============================================================
# CLI
# ============================================================

def parse_head_list(text: str) -> List[int]:
    """
    "2,1" -> [2, 1]
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
        default="./outputs/cifar10_blockwise_fixed_mcnemar",
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

    # Defaults are derived from the immediately preceding diagnostics:
    # B=1:
    #   Block0 prefers H2
    #   Block1 prefers H1
    p.add_argument(
        "--b1-block0",
        type=str,
        default="2",
    )
    p.add_argument(
        "--b1-block1",
        type=str,
        default="1",
    )

    # B=2:
    #   Block0 modal role pair = H2 direct + H1 mixed
    #   Block1 modal role pair = H1 direct + H2 mixed
    p.add_argument(
        "--b2-block0",
        type=str,
        default="2,1",
    )
    p.add_argument(
        "--b2-block1",
        type=str,
        default="1,2",
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
            "This experiment is configured for the current "
            f"depth=2 sanity checkpoint, got depth={len(model.blocks)}."
        )

    if model.main_heads != 3:
        raise ValueError(
            "This experiment expects main_heads=3, "
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

    b1_configs = [
        parse_head_list(args.b1_block0),
        parse_head_list(args.b1_block1),
    ]

    b2_configs = [
        parse_head_list(args.b2_block0),
        parse_head_list(args.b2_block1),
    ]

    if any(len(x) != 1 for x in b1_configs):
        raise ValueError(
            "Each B=1 block config must contain exactly one head."
        )

    if any(len(x) != 2 for x in b2_configs):
        raise ValueError(
            "Each B=2 block config must contain exactly two ordered heads."
        )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Test samples:", len(loader.dataset))
    print(
        "Model:",
        f"depth={len(model.blocks)}, "
        f"main_heads={model.main_heads}, "
        f"img_size={img_size}",
    )
    print()
    print(
        "B=1 fixed:",
        config_to_string(1, b1_configs),
    )
    print(
        "B=2 fixed:",
        config_to_string(2, b2_configs),
    )

    # --------------------------------------------------------
    # B=1
    # --------------------------------------------------------
    learned_b1 = collect_correctness(
        model=model,
        loader=loader,
        device=device,
        budget=1,
    )

    with blockwise_fixed_allocation(
        model=model,
        budget=1,
        block_configs=b1_configs,
    ):
        fixed_b1 = collect_correctness(
            model=model,
            loader=loader,
            device=device,
            budget=1,
        )

    b1_result = report(
        budget=1,
        learned_result=learned_b1,
        fixed_result=fixed_b1,
        block_configs=b1_configs,
    )

    # --------------------------------------------------------
    # B=2
    # --------------------------------------------------------
    learned_b2 = collect_correctness(
        model=model,
        loader=loader,
        device=device,
        budget=2,
    )

    with blockwise_fixed_allocation(
        model=model,
        budget=2,
        block_configs=b2_configs,
    ):
        fixed_b2 = collect_correctness(
            model=model,
            loader=loader,
            device=device,
            budget=2,
        )

    b2_result = report(
        budget=2,
        learned_result=learned_b2,
        fixed_result=fixed_b2,
        block_configs=b2_configs,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "blockwise_fixed_mcnemar.json"
    )

    final = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "B1": b1_result,
        "B2": b2_result,
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
    print("=" * 100)
    print("Block-wise fixed McNemar evaluation finished.")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()