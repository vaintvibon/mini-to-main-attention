import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="/content/cifar10",
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/"
            "checkpoints/budgeted_v2_fair/mini_main/best.pt"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/"
            "checkpoints/budgeted_v2_fair/"
            "b6_b8_routing_consistency"
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


# ============================================================
# Seed
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def build_heldout_dataset(data_dir, seed):

    mean = (
        0.4914,
        0.4822,
        0.4465,
    )

    std = (
        0.2470,
        0.2435,
        0.2616,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    g = torch.Generator().manual_seed(seed)

    perm = torch.randperm(
        len(base),
        generator=g,
    ).tolist()

    # Fair experiment:
    # train   = [0:40000]
    # val     = [40000:45000]
    # heldout = [45000:50000]

    heldout_indices = perm[
        45000:50000
    ]

    return (
        Subset(
            base,
            heldout_indices,
        ),
        heldout_indices,
    )


# ============================================================
# Model
# ============================================================

def load_checkpoint(path, device):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        ckpt = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            path,
            map_location=device,
        )

    return ckpt


def build_model(ckpt, device):

    cfg = ckpt["config"]

    model = BudgetedMiniMainViTV2(
        img_size=cfg.get("img_size", 32),
        patch_size=cfg.get("patch_size", 4),

        num_classes=cfg.get(
            "num_classes",
            10,
        ),

        embed_dim=cfg.get(
            "embed_dim",
            192,
        ),

        depth=cfg.get(
            "depth",
            4,
        ),

        main_heads=cfg.get(
            "main_heads",
            8,
        ),

        mini_heads=cfg.get(
            "mini_heads",
            4,
        ),

        mini_head_dim=cfg.get(
            "mini_head_dim",
            16,
        ),

        direct_k=cfg.get(
            "direct_k",
            2,
        ),

        pool_ratio=cfg.get(
            "pool_ratio",
            2,
        ),

        mode="mini_main",
    ).to(device)

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# Helpers
# ============================================================

def canonical_pair(x):
    """
    [N, 2] direct indices를 unordered pair로 정렬.
    """
    return torch.sort(
        x,
        dim=1,
    ).values


def binding_main_indices(binding_hard):
    """
    binding_hard:
        [N, direct_k, main_heads]

    return:
        [N, direct_k]
    """
    return binding_hard.long().argmax(
        dim=-1
    )


def canonical_binding_map(
    direct_indices,
    binding_hard,
    mini_heads,
):
    """
    예:
       direct rank 0: Mini 3 -> Main 6
       direct rank 1: Mini 1 -> Main 2

    를 Mini 기준 canonical mapping으로 변경.

    output:
       [N, mini_heads]

    선택되지 않은 Mini는 -1.
    """

    n = direct_indices.shape[0]

    main_idx = binding_main_indices(
        binding_hard
    )

    result = torch.full(
        (n, mini_heads),
        -1,
        dtype=torch.long,
    )

    for rank in range(
        direct_indices.shape[1]
    ):

        mini_idx = direct_indices[
            :, rank
        ]

        main = main_idx[
            :, rank
        ]

        result.scatter_(
            1,
            mini_idx[:, None],
            main[:, None],
        )

    return result


def rank_tensor(x):
    """
    각 row에 대해 rank 0..H-1 반환.
    점수가 낮으면 낮은 rank.
    """
    order = torch.argsort(
        x,
        dim=1,
    )

    ranks = torch.empty_like(
        order,
        dtype=torch.float32,
    )

    values = torch.arange(
        x.shape[1],
        dtype=torch.float32,
    )[None].expand(
        x.shape[0],
        -1,
    )

    ranks.scatter_(
        1,
        order,
        values,
    )

    return ranks


def rowwise_spearman(a, b):
    """
    Tie가 거의 없다는 가정 아래
    rank Pearson correlation.
    """

    ra = rank_tensor(a)
    rb = rank_tensor(b)

    ra = ra - ra.mean(
        dim=1,
        keepdim=True,
    )

    rb = rb - rb.mean(
        dim=1,
        keepdim=True,
    )

    numerator = (
        ra * rb
    ).sum(dim=1)

    denominator = (
        torch.sqrt(
            (ra ** 2).sum(dim=1)
        )
        *
        torch.sqrt(
            (rb ** 2).sum(dim=1)
        )
    ).clamp_min(1e-12)

    return numerator / denominator


def counterfactual_b6_mask(
    bound_main_mask,
    need_scores,
    direct_k,
    budget=6,
):
    """
    B8 hidden state에서,
    만약 budget=6 router를 적용했다면
    어떤 Main 6개가 켜졌을지 계산.

    실제 forward는 하지 않고 routing만 재현.
    """

    extra_k = budget - direct_k

    scores = need_scores.clone()

    scores = scores.masked_fill(
        bound_main_mask,
        -float("inf"),
    )

    extra_idx = torch.topk(
        scores,
        k=extra_k,
        dim=1,
    ).indices

    extra_mask = torch.zeros_like(
        bound_main_mask
    )

    extra_mask.scatter_(
        1,
        extra_idx,
        True,
    )

    return (
        bound_main_mask
        |
        extra_mask
    )


def pair_to_string(pair):
    pair = sorted(
        [int(pair[0]), int(pair[1])]
    )

    return f"({pair[0]},{pair[1]})"


# ============================================================
# Evaluate
# ============================================================

@torch.inference_mode()
def collect_budget(
    model,
    loader,
    budget,
    device,
):

    all_targets = []
    all_preds = []

    depth = model.depth

    block_data = [
        {
            "direct_indices": [],
            "binding_hard": [],
            "bound_main_mask": [],
            "active_main_mask": [],
            "need_scores": [],
            "utility_logits": [],
        }
        for _ in range(depth)
    ]

    for batch_idx, (
        images,
        targets,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        logits, infos = model(
            images,
            budget=budget,
            return_info=True,
        )

        preds = logits.argmax(
            dim=1
        )

        all_targets.append(
            targets.cpu()
        )

        all_preds.append(
            preds.cpu()
        )

        for block_idx, info in enumerate(
            infos
        ):

            for key in block_data[
                block_idx
            ].keys():

                block_data[
                    block_idx
                ][key].append(
                    info[key]
                    .detach()
                    .cpu()
                )

        if (
            batch_idx % 10 == 0
            or
            batch_idx + 1 == len(loader)
        ):

            print(
                f"B={budget} "
                f"[{batch_idx + 1}/"
                f"{len(loader)}]"
            )

    for block_idx in range(depth):

        for key in block_data[
            block_idx
        ]:

            block_data[
                block_idx
            ][key] = torch.cat(
                block_data[
                    block_idx
                ][key],
                dim=0,
            )

    return {
        "targets": torch.cat(
            all_targets
        ),

        "preds": torch.cat(
            all_preds
        ),

        "blocks": block_data,
    }


# ============================================================
# Group summary
# ============================================================

def print_group_pair_consistency(
    group_name,
    mask,
    b6,
    b8,
    mini_heads,
):

    print()
    print("=" * 78)
    print(
        f"ROUTING CONSISTENCY — {group_name}"
    )
    print("=" * 78)

    n = int(mask.sum())

    print(
        f"N = {n}"
    )

    if n == 0:
        return

    for block_idx in range(
        len(b6["blocks"])
    ):

        x6 = b6[
            "blocks"
        ][block_idx]

        x8 = b8[
            "blocks"
        ][block_idx]

        d6 = x6[
            "direct_indices"
        ][mask]

        d8 = x8[
            "direct_indices"
        ][mask]

        ordered_same = (
            d6 == d8
        ).all(dim=1)

        set_same = (
            canonical_pair(d6)
            ==
            canonical_pair(d8)
        ).all(dim=1)

        map6 = canonical_binding_map(
            d6,
            x6[
                "binding_hard"
            ][mask],
            mini_heads,
        )

        map8 = canonical_binding_map(
            d8,
            x8[
                "binding_hard"
            ][mask],
            mini_heads,
        )

        mapping_same = (
            map6 == map8
        ).all(dim=1)

        bound_same = (
            x6[
                "bound_main_mask"
            ][mask]
            ==
            x8[
                "bound_main_mask"
            ][mask]
        ).all(dim=1)

        util_diff = (
            x8[
                "utility_logits"
            ][mask]
            -
            x6[
                "utility_logits"
            ][mask]
        ).abs().mean(dim=1)

        need_corr = rowwise_spearman(
            x6[
                "need_scores"
            ][mask],
            x8[
                "need_scores"
            ][mask],
        )

        print()
        print(
            f"Block {block_idx}"
        )

        print(
            f"  Direct pair same (set)   : "
            f"{set_same.float().mean()*100:.2f}%"
        )

        print(
            f"  Direct pair same (order) : "
            f"{ordered_same.float().mean()*100:.2f}%"
        )

        print(
            f"  Mini->Main mapping same  : "
            f"{mapping_same.float().mean()*100:.2f}%"
        )

        print(
            f"  Bound Main set same      : "
            f"{bound_same.float().mean()*100:.2f}%"
        )

        print(
            f"  Utility |Δ| mean         : "
            f"{util_diff.mean():.8f}"
        )

        print(
            f"  Need-score Spearman mean : "
            f"{need_corr.mean():.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "B6 vs B8 BLOCK-WISE ROUTING CONSISTENCY"
    )
    print("=" * 78)

    print(
        f"Device     : {device}"
    )

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    ckpt = load_checkpoint(
        args.checkpoint,
        device,
    )

    model = build_model(
        ckpt,
        device,
    )

    print(
        f"Depth      : {model.depth}"
    )

    print(
        f"Mini heads : {model.mini_heads}"
    )

    print(
        f"Main heads : {model.main_heads}"
    )

    print(
        f"Direct K   : {model.direct_k}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    heldout, heldout_indices = (
        build_heldout_dataset(
            args.data_dir,
            args.seed,
        )
    )

    loader = DataLoader(
        heldout,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Heldout    : {len(heldout)}"
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Collecting B=6")
    print("=" * 78)

    b6 = collect_budget(
        model,
        loader,
        budget=6,
        device=device,
    )

    print()
    print("=" * 78)
    print("Collecting B=8")
    print("=" * 78)

    b8 = collect_budget(
        model,
        loader,
        budget=8,
        device=device,
    )

    if not torch.equal(
        b6["targets"],
        b8["targets"],
    ):
        raise RuntimeError(
            "Target order mismatch"
        )

    # --------------------------------------------------------
    # Prediction groups
    # --------------------------------------------------------

    target = b6[
        "targets"
    ]

    pred6 = b6[
        "preds"
    ]

    pred8 = b8[
        "preds"
    ]

    correct6 = (
        pred6 == target
    )

    correct8 = (
        pred8 == target
    )

    cc = (
        correct6
        &
        correct8
    )

    degraded = (
        correct6
        &
        ~correct8
    )

    rescued = (
        ~correct6
        &
        correct8
    )

    ww = (
        ~correct6
        &
        ~correct8
    )

    print()
    print("=" * 78)
    print("PREDICTION GROUPS")
    print("=" * 78)

    print(
        f"Correct -> Correct : {cc.sum().item()}"
    )

    print(
        f"Correct -> Wrong   : {degraded.sum().item()}"
    )

    print(
        f"Wrong -> Correct   : {rescued.sum().item()}"
    )

    print(
        f"Wrong -> Wrong     : {ww.sum().item()}"
    )

    # Expected from previous experiment:
    #
    # CC = 3725
    # CW = 81
    # WC = 70
    # WW = 1124

    # --------------------------------------------------------
    # Overall block-wise analysis
    # --------------------------------------------------------

    summary_rows = []

    n = len(target)

    print()
    print("=" * 78)
    print("OVERALL BLOCK-WISE ROUTING")
    print("=" * 78)

    for block_idx in range(
        model.depth
    ):

        x6 = b6[
            "blocks"
        ][block_idx]

        x8 = b8[
            "blocks"
        ][block_idx]

        direct6 = x6[
            "direct_indices"
        ]

        direct8 = x8[
            "direct_indices"
        ]

        # ----------------------------------------------------
        # Direct pair
        # ----------------------------------------------------

        pair_order_same = (
            direct6 == direct8
        ).all(dim=1)

        pair_set_same = (
            canonical_pair(
                direct6
            )
            ==
            canonical_pair(
                direct8
            )
        ).all(dim=1)

        # ----------------------------------------------------
        # Binding
        # ----------------------------------------------------

        map6 = canonical_binding_map(
            direct6,
            x6["binding_hard"],
            model.mini_heads,
        )

        map8 = canonical_binding_map(
            direct8,
            x8["binding_hard"],
            model.mini_heads,
        )

        mapping_same = (
            map6 == map8
        ).all(dim=1)

        bound_set_same = (
            x6["bound_main_mask"]
            ==
            x8["bound_main_mask"]
        ).all(dim=1)

        # ----------------------------------------------------
        # Utility changes
        # ----------------------------------------------------

        utility_abs_diff = (
            x8["utility_logits"]
            -
            x6["utility_logits"]
        ).abs().mean(dim=1)

        utility_spearman = (
            rowwise_spearman(
                x6["utility_logits"],
                x8["utility_logits"],
            )
        )

        # ----------------------------------------------------
        # Need scores
        # ----------------------------------------------------

        need_abs_diff = (
            x8["need_scores"]
            -
            x6["need_scores"]
        ).abs().mean(dim=1)

        need_spearman = (
            rowwise_spearman(
                x6["need_scores"],
                x8["need_scores"],
            )
        )

        # ----------------------------------------------------
        # Counterfactual:
        # B8 state에서 budget=6이라면
        # 어떤 active heads?
        # ----------------------------------------------------

        b8_as_b6 = (
            counterfactual_b6_mask(
                x8[
                    "bound_main_mask"
                ],
                x8[
                    "need_scores"
                ],
                model.direct_k,
                budget=6,
            )
        )

        cf_b6_same = (
            x6[
                "active_main_mask"
            ]
            ==
            b8_as_b6
        ).all(dim=1)

        # ----------------------------------------------------
        # B6 excluded head frequency
        # ----------------------------------------------------

        excluded6 = (
            ~x6[
                "active_main_mask"
            ]
        )

        print()
        print(
            f"---------------- BLOCK {block_idx} ----------------"
        )

        print(
            f"Direct pair same [set]     : "
            f"{pair_set_same.float().mean()*100:.2f}%"
        )

        print(
            f"Direct pair same [ordered] : "
            f"{pair_order_same.float().mean()*100:.2f}%"
        )

        print(
            f"Mini->Main mapping same    : "
            f"{mapping_same.float().mean()*100:.2f}%"
        )

        print(
            f"Bound Main set same        : "
            f"{bound_set_same.float().mean()*100:.2f}%"
        )

        print(
            f"Utility |Δ| mean           : "
            f"{utility_abs_diff.mean():.8f}"
        )

        print(
            f"Utility rank Spearman      : "
            f"{utility_spearman.mean():.4f}"
        )

        print(
            f"Need-score |Δ| mean        : "
            f"{need_abs_diff.mean():.8f}"
        )

        print(
            f"Need-score rank Spearman   : "
            f"{need_spearman.mean():.4f}"
        )

        print(
            f"B8-state virtual B6 same   : "
            f"{cf_b6_same.float().mean()*100:.2f}%"
        )

        print()
        print(
            "B6 excluded Main head frequency:"
        )

        for h in range(
            model.main_heads
        ):

            freq = (
                excluded6[:, h]
                .float()
                .mean()
                .item()
                * 100
            )

            print(
                f"  H{h}: {freq:6.2f}%"
            )

        summary_rows.append({
            "block":
                block_idx,

            "direct_pair_set_same_pct":
                pair_set_same.float()
                .mean()
                .item()
                * 100,

            "direct_pair_order_same_pct":
                pair_order_same.float()
                .mean()
                .item()
                * 100,

            "binding_mapping_same_pct":
                mapping_same.float()
                .mean()
                .item()
                * 100,

            "bound_main_set_same_pct":
                bound_set_same.float()
                .mean()
                .item()
                * 100,

            "utility_abs_diff_mean":
                utility_abs_diff.mean()
                .item(),

            "utility_rank_spearman":
                utility_spearman.mean()
                .item(),

            "need_abs_diff_mean":
                need_abs_diff.mean()
                .item(),

            "need_rank_spearman":
                need_spearman.mean()
                .item(),

            "virtual_b6_route_same_pct":
                cf_b6_same.float()
                .mean()
                .item()
                * 100,
        })

    # --------------------------------------------------------
    # Prediction-group analysis
    # --------------------------------------------------------

    print_group_pair_consistency(
        "ALL",
        torch.ones(
            n,
            dtype=torch.bool,
        ),
        b6,
        b8,
        model.mini_heads,
    )

    print_group_pair_consistency(
        "CORRECT -> CORRECT",
        cc,
        b6,
        b8,
        model.mini_heads,
    )

    print_group_pair_consistency(
        "CORRECT -> WRONG [DEGRADED]",
        degraded,
        b6,
        b8,
        model.mini_heads,
    )

    print_group_pair_consistency(
        "WRONG -> CORRECT [RESCUED]",
        rescued,
        b6,
        b8,
        model.mini_heads,
    )

    print_group_pair_consistency(
        "WRONG -> WRONG",
        ww,
        b6,
        b8,
        model.mini_heads,
    )

    # ========================================================
    # Per-sample CSV
    # ========================================================

    rows = []

    for i in range(n):

        if cc[i]:
            pred_group = "correct_correct"

        elif degraded[i]:
            pred_group = "correct_wrong"

        elif rescued[i]:
            pred_group = "wrong_correct"

        else:
            pred_group = "wrong_wrong"

        row = {
            "heldout_position": i,

            "original_cifar_index":
                heldout_indices[i],

            "target":
                int(target[i]),

            "b6_pred":
                int(pred6[i]),

            "b8_pred":
                int(pred8[i]),

            "prediction_group":
                pred_group,
        }

        for block_idx in range(
            model.depth
        ):

            x6 = b6[
                "blocks"
            ][block_idx]

            x8 = b8[
                "blocks"
            ][block_idx]

            d6 = x6[
                "direct_indices"
            ][i:i+1]

            d8 = x8[
                "direct_indices"
            ][i:i+1]

            set_same = bool(
                (
                    canonical_pair(d6)
                    ==
                    canonical_pair(d8)
                ).all()
            )

            ordered_same = bool(
                (d6 == d8).all()
            )

            map6 = canonical_binding_map(
                d6,
                x6[
                    "binding_hard"
                ][i:i+1],
                model.mini_heads,
            )

            map8 = canonical_binding_map(
                d8,
                x8[
                    "binding_hard"
                ][i:i+1],
                model.mini_heads,
            )

            mapping_same = bool(
                (map6 == map8).all()
            )

            need_corr = (
                rowwise_spearman(
                    x6[
                        "need_scores"
                    ][i:i+1],
                    x8[
                        "need_scores"
                    ][i:i+1],
                )[0].item()
            )

            row[
                f"block{block_idx}_b6_pair"
            ] = pair_to_string(
                d6[0].tolist()
            )

            row[
                f"block{block_idx}_b8_pair"
            ] = pair_to_string(
                d8[0].tolist()
            )

            row[
                f"block{block_idx}_pair_set_same"
            ] = int(set_same)

            row[
                f"block{block_idx}_pair_order_same"
            ] = int(ordered_same)

            row[
                f"block{block_idx}_binding_same"
            ] = int(mapping_same)

            row[
                f"block{block_idx}_need_rank_corr"
            ] = need_corr

        rows.append(row)

    df = pd.DataFrame(rows)

    # Any downstream change:
    # Block 1~3 중 하나라도 direct pair가 달라졌는지.

    downstream_cols = [
        f"block{i}_pair_set_same"
        for i in range(
            1,
            model.depth,
        )
    ]

    if downstream_cols:

        df[
            "any_downstream_pair_change"
        ] = (
            df[
                downstream_cols
            ].min(axis=1)
            == 0
        ).astype(int)

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

    summary_path = (
        output_dir
        /
        "b6_b8_routing_block_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
    )

    sample_path = (
        output_dir
        /
        "b6_b8_routing_per_sample.csv"
    )

    df.to_csv(
        sample_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final sanity
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("BLOCK 0 SANITY")
    print("=" * 78)

    block0 = summary_rows[0]

    print(
        "Because B6 and B8 enter Block 0 "
        "with the SAME hidden state:"
    )

    print(
        f"Direct pair same          : "
        f"{block0['direct_pair_set_same_pct']:.2f}%"
    )

    print(
        f"Binding mapping same      : "
        f"{block0['binding_mapping_same_pct']:.2f}%"
    )

    print(
        f"Utility |Δ|              : "
        f"{block0['utility_abs_diff_mean']:.10f}"
    )

    if (
        block0[
            "direct_pair_set_same_pct"
        ] < 99.99
    ):

        print()
        print(
            "[WARNING] Block 0 direct pair "
            "should normally be identical."
        )

    # --------------------------------------------------------
    # Divergence concentration
    # --------------------------------------------------------

    if downstream_cols:

        print()
        print("=" * 78)
        print(
            "DOWNSTREAM DIRECT-PAIR DIVERGENCE"
        )
        print("=" * 78)

        for group in [
            "correct_correct",
            "correct_wrong",
            "wrong_correct",
            "wrong_wrong",
        ]:

            subset = df[
                df[
                    "prediction_group"
                ] == group
            ]

            if len(subset) == 0:
                continue

            rate = (
                subset[
                    "any_downstream_pair_change"
                ].mean()
                * 100
            )

            print(
                f"{group:<18}: "
                f"{rate:6.2f}% "
                f"({int(subset['any_downstream_pair_change'].sum())}"
                f"/{len(subset)})"
            )

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)

    print(summary_path)
    print(sample_path)

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        """
1. Block 0에서는 B6/B8 입력이 동일하므로
   Direct pair와 binding이 사실상 100% 같아야 한다.

2. Block 1 -> 2 -> 3에서
   Direct-pair same rate가 빠르게 떨어진다면:

   B6 vs B8 차이는 단순히
   'Main head 두 개 추가'가 아니다.

   초기 Main 계산 차이
       ↓
   다음 block hidden state 변화
       ↓
   Mini utility 변화
       ↓
   Direct pair / binding 변화
       ↓
   routing divergence 누적

   으로 해석할 수 있다.

3. 반대로 Block 1~3에서도
   pair/binding consistency가 거의 100%라면:

   B6-B8 accuracy 차이는
   routing cascade 때문이라고 보기 어렵다.

   그러면 다음은
   B6에서 제외된 Main 2개의
   marginal contribution ablation이다.

4. 특히 CORRECT->WRONG 81개에서
   downstream pair-change rate가
   전체보다 훨씬 높다면:

   B8의 추가 computation이
   downstream routing을 바꾸면서
   일부 boundary sample을 악화시키는
   가설이 강해진다.

5. CORRECT->WRONG과 WRONG->CORRECT에서
   routing-change rate가 비슷하다면:

   routing divergence 자체는 존재하지만
   그것이 일방적으로 harmful하다고 볼 수 없다.
        """
    )


if __name__ == "__main__":
    main()