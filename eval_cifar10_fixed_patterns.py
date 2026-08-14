# eval_cifar10_fixed_patterns.py
"""
Post-hoc diagnostic evaluation for a trained Mini-to-Main Attention checkpoint.

재학습 없이 best.pt를 로드해서 다음을 평가한다.

1) B=1 fixed single-head 전체 비교
   - H0
   - H1
   - H2

2) B=2 fixed pair 전체 비교
   direct/mixed 역할이 다르므로 unordered pair 하나당 두 방향을 모두 평가한다.
   - H0(direct) -> H1(mixed)
   - H1(direct) -> H0(mixed)
   - H0(direct) -> H2(mixed)
   - H2(direct) -> H0(mixed)
   - H1(direct) -> H2(mixed)
   - H2(direct) -> H1(mixed)

   그리고 unordered pair별 mean / best orientation을 따로 요약한다.

3) Learned allocation의 block별 selection frequency
   - B=1
   - B=2

4) Learned allocation의 sample-level cross-block pattern distribution
   예: depth=2, B=1
       Block0:H1 | Block1:H2
   같은 sample 단위 pattern을 집계한다.

핵심 질문
---------
A. Learned B=1이 Best Fixed Single보다 좋은가?
B. Learned B=2가 Best Fixed Pair/Role보다 좋은가?
C. 같은 block 안에서도 여러 head/pair가 sample에 따라 선택되는가?
D. block별 고정 specialization만 있는 것이 아니라 sample-dependent selection이 있는가?
"""

import argparse
import json
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
# Basic utility
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
            f"Missing model keys while loading checkpoint: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected model keys while loading checkpoint: {unexpected}"
        )

    model.eval()

    return model


def build_test_loader(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
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

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader


# ============================================================
# Fixed scheduler override
# ============================================================

def _build_fixed_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    ordered_heads: Sequence[int],
):
    """
    ordered_heads의 앞쪽이 direct 우선순위를 갖는다.

    B=1, ordered_heads=[2]
        -> H2 direct

    B=2, ordered_heads=[2,0]
        -> H2 direct, H0 mixed
        (direct_ratio=0.34 / H=3 기준 direct_count=1)
    """
    B, H = alloc_logits.shape

    if budget < 0 or budget > H:
        raise ValueError(
            f"Invalid budget={budget} for H={H}."
        )

    if len(ordered_heads) < budget:
        raise ValueError(
            "ordered_heads is shorter than budget."
        )

    if len(set(ordered_heads[:budget])) != budget:
        raise ValueError(
            "Duplicate heads in fixed configuration."
        )

    for h in ordered_heads[:budget]:
        if h < 0 or h >= H:
            raise ValueError(
                f"Invalid head index H{h}; main_heads={H}."
            )

    direct_count = scheduler._get_direct_count(
        budget
    )

    selected = torch.tensor(
        list(ordered_heads[:budget]),
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
        direct_idx = selected[:, :direct_count]
        direct_mask.scatter_(
            1,
            direct_idx,
            True,
        )

    mixed_mask = active_mask & (~direct_mask)
    inactive_mask = ~active_mask

    dtype = alloc_logits.dtype

    # eval 전용: gate는 정확한 hard mask
    active_gate = active_mask.to(dtype)
    direct_gate = direct_mask.to(dtype)
    mixed_gate = mixed_mask.to(dtype)
    inactive_gate = inactive_mask.to(dtype)

    # selection_scores는 scheduler 출력 인터페이스 호환용.
    # 높은 순위가 ordered_heads 앞쪽에 오도록 만든다.
    rank_scores = torch.zeros_like(
        alloc_logits
    )

    for rank, head in enumerate(
        ordered_heads[:budget]
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
    """
    모든 transformer block의 scheduler를 동일 fixed rule로 임시 교체.
    context 종료 시 원래 learned scheduler 복원.
    """
    original_forwards = []

    for block in model.blocks:
        scheduler = block.attn.scheduler

        original_forwards.append(
            (scheduler, scheduler.forward)
        )

        def override_forward(
            self_scheduler,
            alloc_logits,
            runtime_budget,
            *,
            _budget=budget,
            _ordered_heads=tuple(ordered_heads),
        ):
            if runtime_budget != _budget:
                raise ValueError(
                    f"Expected budget={_budget}, "
                    f"got {runtime_budget}."
                )

            return _build_fixed_schedule(
                scheduler=self_scheduler,
                alloc_logits=alloc_logits,
                budget=runtime_budget,
                ordered_heads=_ordered_heads,
            )

        scheduler.forward = MethodType(
            override_forward,
            scheduler,
        )

    try:
        yield
    finally:
        for scheduler, original in original_forwards:
            scheduler.forward = original


# ============================================================
# Accuracy evaluation
# ============================================================

@torch.no_grad()
def evaluate_accuracy(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

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

        # 혹시 구현상 tuple 반환이면 첫 요소 사용.
        if isinstance(logits, tuple):
            logits = logits[0]

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

    return {
        "accuracy": 100.0 * correct / total,
        "loss": loss_sum / total,
        "samples": total,
    }


@torch.no_grad()
def evaluate_fixed_configuration(
    model,
    loader,
    device,
    budget,
    ordered_heads,
):
    with fixed_allocation(
        model=model,
        budget=budget,
        ordered_heads=ordered_heads,
    ):
        return evaluate_accuracy(
            model=model,
            loader=loader,
            device=device,
            budget=budget,
        )


# ============================================================
# Learned block/sample pattern analysis
# ============================================================

def mask_to_head_tuple(
    mask_row: torch.Tensor,
) -> Tuple[int, ...]:
    idx = torch.nonzero(
        mask_row,
        as_tuple=False,
    ).flatten().tolist()

    return tuple(int(x) for x in idx)


def format_selected_heads(
    heads: Tuple[int, ...],
) -> str:
    if len(heads) == 0:
        return "none"

    return "+".join(
        f"H{h}" for h in heads
    )


@torch.no_grad()
def analyze_learned_patterns(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
    top_k_patterns: int,
):
    """
    learned deterministic eval scheduler를 그대로 사용.

    block_head_counts[block][head]:
        해당 block에서 각 head가 선택된 sample 수.

    block_mask_patterns[block]:
        해당 block 내부에서 active subset pattern 빈도.

    cross_block_patterns:
        sample 하나가 모든 block에서 어떤 active subset을 선택했는지
        joint pattern을 집계.
    """
    model.eval()

    depth = len(model.blocks)
    H = model.main_heads

    block_head_counts = [
        torch.zeros(
            H,
            dtype=torch.long,
        )
        for _ in range(depth)
    ]

    block_mask_patterns = [
        Counter()
        for _ in range(depth)
    ]

    cross_block_patterns = Counter()

    total_samples = 0

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        _, info_list = model(
            images,
            budget=budget,
            return_info=True,
        )

        if len(info_list) != depth:
            raise RuntimeError(
                f"Expected {depth} block infos, "
                f"got {len(info_list)}."
            )

        batch_size = (
            info_list[0]["active_mask"]
            .shape[0]
        )

        batch_block_head_tuples = []

        for block_idx, info in enumerate(
            info_list
        ):
            active = (
                info["active_mask"]
                .detach()
                .cpu()
                .bool()
            )

            block_head_counts[
                block_idx
            ] += active.long().sum(dim=0)

            sample_tuples = []

            for row in active:
                heads = mask_to_head_tuple(row)
                sample_tuples.append(heads)

                label = format_selected_heads(
                    heads
                )
                block_mask_patterns[
                    block_idx
                ][label] += 1

            batch_block_head_tuples.append(
                sample_tuples
            )

        # sample별 cross-block joint pattern
        for sample_idx in range(batch_size):
            parts = []

            for block_idx in range(depth):
                heads = (
                    batch_block_head_tuples[
                        block_idx
                    ][sample_idx]
                )

                parts.append(
                    f"Block{block_idx}:"
                    f"{format_selected_heads(heads)}"
                )

            pattern = " | ".join(parts)
            cross_block_patterns[pattern] += 1

        total_samples += batch_size

    block_summaries = []

    for block_idx in range(depth):
        head_counts = block_head_counts[
            block_idx
        ]

        head_freq_pct = [
            100.0 * int(count) / total_samples
            for count in head_counts
        ]

        mask_top = []

        for pattern, count in (
            block_mask_patterns[
                block_idx
            ].most_common(top_k_patterns)
        ):
            mask_top.append(
                {
                    "pattern": pattern,
                    "count": count,
                    "pct": (
                        100.0
                        * count
                        / total_samples
                    ),
                }
            )

        # 가장 많이 선택된 단일 head의 비율.
        # B=2에서는 각 sample마다 2개가 선택되므로
        # collapse 의미보다는 slot dominance 지표로만 본다.
        max_head_freq = max(
            head_freq_pct
        )

        block_summaries.append(
            {
                "block": block_idx,
                "head_selection_freq_pct": (
                    head_freq_pct
                ),
                "top_subset_patterns": mask_top,
                "max_head_selection_pct": (
                    max_head_freq
                ),
            }
        )

    cross_top = []

    for pattern, count in (
        cross_block_patterns
        .most_common(top_k_patterns)
    ):
        cross_top.append(
            {
                "pattern": pattern,
                "count": count,
                "pct": (
                    100.0
                    * count
                    / total_samples
                ),
            }
        )

    # sample-dependent diversity 진단:
    # 각 block에서 실제로 몇 종류의 subset이 등장했는지.
    per_block_unique_patterns = [
        len(counter)
        for counter in block_mask_patterns
    ]

    return {
        "budget": budget,
        "samples": total_samples,
        "blocks": block_summaries,
        "per_block_unique_patterns": (
            per_block_unique_patterns
        ),
        "cross_block_unique_patterns": len(
            cross_block_patterns
        ),
        "top_cross_block_patterns": cross_top,
    }


# ============================================================
# Printing
# ============================================================

def print_fixed_b1(
    learned_result,
    single_results,
):
    print()
    print("=" * 92)
    print("B=1: Learned vs every fixed single head")
    print("=" * 92)

    print(
        f"Learned     acc={learned_result['accuracy']:.2f}% "
        f"loss={learned_result['loss']:.4f}"
    )

    for head in sorted(single_results):
        r = single_results[head]

        print(
            f"Fixed H{head:<2}  "
            f"acc={r['accuracy']:.2f}% "
            f"loss={r['loss']:.4f} | "
            f"learned-fixed="
            f"{learned_result['accuracy'] - r['accuracy']:+.2f}pp"
        )

    best_head = max(
        single_results,
        key=lambda h: single_results[h][
            "accuracy"
        ],
    )

    best = single_results[best_head]

    print()
    print(
        f"Best Fixed Single: H{best_head} "
        f"({best['accuracy']:.2f}%)"
    )
    print(
        f"Learned - Best Fixed Single = "
        f"{learned_result['accuracy'] - best['accuracy']:+.2f}pp"
    )


def print_fixed_b2(
    learned_result,
    ordered_results,
    pair_summary,
):
    print()
    print("=" * 92)
    print(
        "B=2: Learned vs every fixed pair "
        "(direct/mixed role order included)"
    )
    print("=" * 92)

    print(
        f"Learned     acc={learned_result['accuracy']:.2f}% "
        f"loss={learned_result['loss']:.4f}"
    )

    print()
    print("Role-specific fixed configurations:")

    for key, r in ordered_results.items():
        direct, mixed = key

        print(
            f"  H{direct}(direct) + "
            f"H{mixed}(mixed) | "
            f"acc={r['accuracy']:.2f}% "
            f"loss={r['loss']:.4f} | "
            f"learned-fixed="
            f"{learned_result['accuracy'] - r['accuracy']:+.2f}pp"
        )

    print()
    print("Unordered pair summary:")

    for pair_key, summary in pair_summary.items():
        a, b = pair_key

        print(
            f"  H{a}+H{b} | "
            f"mean={summary['mean_accuracy']:.2f}% | "
            f"best={summary['best_accuracy']:.2f}% "
            f"({summary['best_orientation']}) | "
            f"learned-best="
            f"{learned_result['accuracy'] - summary['best_accuracy']:+.2f}pp"
        )

    best_key = max(
        ordered_results,
        key=lambda key: ordered_results[
            key
        ]["accuracy"],
    )

    best = ordered_results[best_key]

    print()
    print(
        "Best Fixed Pair/Role: "
        f"H{best_key[0]}(direct)+"
        f"H{best_key[1]}(mixed) "
        f"({best['accuracy']:.2f}%)"
    )

    print(
        f"Learned - Best Fixed Pair/Role = "
        f"{learned_result['accuracy'] - best['accuracy']:+.2f}pp"
    )


def print_pattern_analysis(
    analysis,
    top_k_patterns,
):
    budget = analysis["budget"]

    print()
    print("=" * 92)
    print(
        f"B={budget}: Learned block-wise / "
        "sample-wise selection patterns"
    )
    print("=" * 92)

    print(
        f"Samples: {analysis['samples']}"
    )

    for block in analysis["blocks"]:
        print()
        print(
            f"Block {block['block']}"
        )

        freq = (
            block[
                "head_selection_freq_pct"
            ]
        )

        print(
            "  head frequency: "
            + " ".join(
                f"H{i}:{v:.2f}%"
                for i, v in enumerate(freq)
            )
        )

        print(
            "  unique subset patterns: "
            f"{analysis['per_block_unique_patterns'][block['block']]}"
        )

        print(
            "  top subset patterns:"
        )

        for item in (
            block["top_subset_patterns"]
        ):
            print(
                f"    {item['pattern']:<12} "
                f"{item['pct']:6.2f}% "
                f"({item['count']})"
            )

    print()
    print(
        "Cross-block unique sample patterns: "
        f"{analysis['cross_block_unique_patterns']}"
    )

    print(
        f"Top {top_k_patterns} cross-block patterns:"
    )

    for item in (
        analysis[
            "top_cross_block_patterns"
        ]
    ):
        print(
            f"  {item['pct']:6.2f}% "
            f"({item['count']:5d}) | "
            f"{item['pattern']}"
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
        help="Path to trained best.pt",
    )

    p.add_argument(
        "--data-dir",
        type=str,
        default="./datasets",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/cifar10_fixed_pattern_eval",
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
        "--top-k-patterns",
        type=int,
        default=20,
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

    checkpoint_args = checkpoint.get(
        "args",
        {},
    )

    img_size = cfg_value(
        checkpoint_args,
        "img_size",
        224,
    )

    if model.main_heads != 3:
        raise ValueError(
            "This diagnostic script currently expects "
            f"main_heads=3, got {model.main_heads}."
        )

    loader = build_test_loader(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    # --------------------------------------------------------
    # Learned reference
    # --------------------------------------------------------
    learned_b1 = evaluate_accuracy(
        model,
        loader,
        device,
        budget=1,
    )

    learned_b2 = evaluate_accuracy(
        model,
        loader,
        device,
        budget=2,
    )

    # --------------------------------------------------------
    # ① B=1: H0 / H1 / H2
    # --------------------------------------------------------
    single_results = {}

    for head in range(3):
        single_results[head] = (
            evaluate_fixed_configuration(
                model=model,
                loader=loader,
                device=device,
                budget=1,
                ordered_heads=[head],
            )
        )

    # --------------------------------------------------------
    # ② B=2: all 3 pairs, both role orientations
    # --------------------------------------------------------
    ordered_pair_configs = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ]

    ordered_pair_results = {}

    for direct_head, mixed_head in (
        ordered_pair_configs
    ):
        ordered_pair_results[
            (direct_head, mixed_head)
        ] = evaluate_fixed_configuration(
            model=model,
            loader=loader,
            device=device,
            budget=2,
            ordered_heads=[
                direct_head,
                mixed_head,
            ],
        )

    unordered_pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    pair_summary = {}

    for a, b in unordered_pairs:
        r_ab = ordered_pair_results[
            (a, b)
        ]
        r_ba = ordered_pair_results[
            (b, a)
        ]

        if (
            r_ab["accuracy"]
            >= r_ba["accuracy"]
        ):
            best_orientation = (
                f"H{a}(direct)+"
                f"H{b}(mixed)"
            )
            best_accuracy = (
                r_ab["accuracy"]
            )
        else:
            best_orientation = (
                f"H{b}(direct)+"
                f"H{a}(mixed)"
            )
            best_accuracy = (
                r_ba["accuracy"]
            )

        pair_summary[(a, b)] = {
            "mean_accuracy": (
                r_ab["accuracy"]
                + r_ba["accuracy"]
            ) / 2.0,
            "best_accuracy": (
                best_accuracy
            ),
            "best_orientation": (
                best_orientation
            ),
        }

    # --------------------------------------------------------
    # ③ learned block/sample patterns
    # --------------------------------------------------------
    pattern_b1 = (
        analyze_learned_patterns(
            model=model,
            loader=loader,
            device=device,
            budget=1,
            top_k_patterns=(
                args.top_k_patterns
            ),
        )
    )

    pattern_b2 = (
        analyze_learned_patterns(
            model=model,
            loader=loader,
            device=device,
            budget=2,
            top_k_patterns=(
                args.top_k_patterns
            ),
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print_fixed_b1(
        learned_result=learned_b1,
        single_results=single_results,
    )

    print_fixed_b2(
        learned_result=learned_b2,
        ordered_results=(
            ordered_pair_results
        ),
        pair_summary=pair_summary,
    )

    print_pattern_analysis(
        pattern_b1,
        args.top_k_patterns,
    )

    print_pattern_analysis(
        pattern_b2,
        args.top_k_patterns,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    json_single = {
        f"H{head}": result
        for head, result
        in single_results.items()
    }

    json_ordered_pairs = {
        (
            f"H{direct}_direct__"
            f"H{mixed}_mixed"
        ): result
        for (direct, mixed), result
        in ordered_pair_results.items()
    }

    json_pair_summary = {
        f"H{a}+H{b}": summary
        for (a, b), summary
        in pair_summary.items()
    }

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
        "learned": {
            "B1": learned_b1,
            "B2": learned_b2,
        },
        "fixed_B1_single_heads": (
            json_single
        ),
        "fixed_B2_ordered_roles": (
            json_ordered_pairs
        ),
        "fixed_B2_pair_summary": (
            json_pair_summary
        ),
        "learned_pattern_analysis": {
            "B1": pattern_b1,
            "B2": pattern_b2,
        },
    }

    output_path = (
        output_dir
        / "fixed_and_pattern_analysis.json"
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
    print("=" * 92)
    print("Diagnostic evaluation finished.")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
