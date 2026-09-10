import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "budgeted_v2_fair/mini_main/best.pt"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "budgeted_v2_fair/"
            "b6_excluded_head_marginal"
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--skip-isolated",
        action="store_true",
        help="Skip block-by-block intervention experiments.",
    )

    return p.parse_args()


# ============================================================
# Reproducibility
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
        50000,
        generator=g,
    ).tolist()

    heldout_indices = perm[45000:50000]

    return (
        Subset(base, heldout_indices),
        heldout_indices,
    )


# ============================================================
# Model
# ============================================================

def load_checkpoint(path, device):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def build_model(ckpt, device):

    cfg = ckpt.get("config", {})

    model = BudgetedMiniMainViTV2(
        img_size=cfg.get("img_size", 32),
        patch_size=cfg.get("patch_size", 4),
        in_chans=3,

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
# Custom B6 + excluded head routing
# ============================================================

def make_custom_active_mask(
    baseline_mask,
    need_scores,
    mode,
):
    """
    baseline_mask:
        B6 active mask [B, 8].

    Since B6 activates exactly 6 heads,
    exactly 2 Main heads are excluded.

    NEXT:
        Add the excluded head with HIGHER need score.

    LAST:
        Add the excluded head with LOWER need score.

    BOTH:
        Add both excluded heads.
    """

    B, H = baseline_mask.shape

    excluded = ~baseline_mask

    excluded_count = excluded.sum(dim=1)

    if not torch.all(
        excluded_count == 2
    ):
        raise RuntimeError(
            "Expected exactly two excluded heads "
            "under B6."
        )

    if mode == "NEXT":

        scores = need_scores.masked_fill(
            ~excluded,
            -float("inf"),
        )

        added = scores.argmax(
            dim=1
        )

        active = baseline_mask.clone()

        active.scatter_(
            1,
            added[:, None],
            True,
        )

        return (
            active,
            added[:, None],
            7,
        )

    if mode == "LAST":

        scores = need_scores.masked_fill(
            ~excluded,
            float("inf"),
        )

        added = scores.argmin(
            dim=1
        )

        active = baseline_mask.clone()

        active.scatter_(
            1,
            added[:, None],
            True,
        )

        return (
            active,
            added[:, None],
            7,
        )

    if mode == "BOTH":

        active = torch.ones_like(
            baseline_mask
        )

        # [B*2, 2] -> [B,2]
        excluded_idx = torch.nonzero(
            excluded,
            as_tuple=False,
        )[:, 1].reshape(B, 2)

        return (
            active,
            excluded_idx,
            8,
        )

    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# Custom attention forward
# ============================================================

def custom_attention_forward(
    attn,
    x,
    patch_hw,
    add_mode,
):
    """
    Important:
    1) Compute Mini and seeds exactly as B6.
    2) Get B6 router mask.
    3) Add one/both of the B6-excluded heads.
    4) Run Main with modified active mask.

    No weights are changed.
    """

    B, N, _ = x.shape

    # --------------------------------------------------------
    # Mini
    # --------------------------------------------------------

    mini_contexts, mini_attn = (
        attn.mini(
            x,
            patch_hw,
        )
    )

    utility_logits = attn.utility(
        mini_contexts,
        mini_attn,
    )

    mini_cat = (
        mini_contexts
        .transpose(1, 2)
        .reshape(
            B,
            N,
            attn.mini_heads
            * attn.mini_head_dim,
        )
    )

    mini_base = attn.mini_base_proj(
        mini_cat
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Always obtain routing/seeds from B6.
    # --------------------------------------------------------

    seeds, route_info = attn.router(
        mini_contexts,
        utility_logits,
        budget=6,
        training_st=False,
    )

    b6_mask = route_info[
        "active_main_mask"
    ]

    need_scores = route_info[
        "need_scores"
    ]

    # --------------------------------------------------------
    # Add excluded Main head(s)
    # --------------------------------------------------------

    (
        custom_mask,
        added_heads,
        effective_budget,
    ) = make_custom_active_mask(
        b6_mask,
        need_scores,
        add_mode,
    )

    custom_gate = custom_mask.to(
        dtype=x.dtype
    )

    # --------------------------------------------------------
    # Main attention
    # --------------------------------------------------------

    main_out = attn.main(
        x,
        seeds,
        custom_mask,
        custom_gate,
        effective_budget,
        False,   # force_dense
        False,   # return_info
    )

    out = (
        attn.mini_base_scale
        * mini_base
        + main_out
    )

    info = {
        "b6_active_mask":
            b6_mask.detach(),

        "custom_active_mask":
            custom_mask.detach(),

        "added_heads":
            added_heads.detach(),

        "need_scores":
            need_scores.detach(),
    }

    return out, info


# ============================================================
# Custom block
# ============================================================

def custom_block_forward(
    block,
    x,
    patch_hw,
    add_mode,
):

    z = block.norm1(x)

    attn_out, info = (
        custom_attention_forward(
            block.attn,
            z,
            patch_hw,
            add_mode,
        )
    )

    x = x + block.dp1(
        attn_out
    )

    x = x + block.dp2(
        block.mlp(
            block.norm2(x)
        )
    )

    return x, info


# ============================================================
# Custom ViT forward
# ============================================================

def custom_model_forward(
    model,
    images,
    add_mode,
    target_block=None,
):
    """
    target_block=None:
        Intervention at ALL blocks.

    target_block=i:
        Only Block i gets custom routing.
        All other blocks use ordinary B6.

    This lets us answer both:

    - What happens if we add next/last head globally?
    - Which block causes the effect?
    """

    B = images.shape[0]

    x = model.patch_embed(
        images
    )

    x = torch.cat(
        [
            model.cls_token.expand(
                B,
                -1,
                -1,
            ),
            x,
        ],
        dim=1,
    )

    x = model.pos_drop(
        x + model.pos_embed
    )

    intervention_infos = []

    for block_idx, block in enumerate(
        model.blocks
    ):

        intervene = (
            target_block is None
            or
            block_idx == target_block
        )

        if intervene:

            x, info = custom_block_forward(
                block,
                x,
                model.patch_hw,
                add_mode,
            )

            intervention_infos.append(
                (
                    block_idx,
                    info,
                )
            )

        else:

            # All non-intervened blocks stay at B6.
            x = block(
                x,
                budget=6,
                patch_hw=model.patch_hw,
                force_dense_main=False,
                return_info=False,
            )

    cls = model.norm(x)[:, 0]

    logits = model.head(cls)

    return (
        logits,
        intervention_infos,
    )


# ============================================================
# Per sample metrics
# ============================================================

def stats_from_logits(
    logits,
    targets,
):

    loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    pred = logits.argmax(
        dim=1
    )

    correct = (
        pred == targets
    )

    return {
        "logits": logits.detach().cpu(),
        "loss": loss.detach().cpu(),
        "pred": pred.detach().cpu(),
        "correct": correct.detach().cpu(),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_condition(
    model,
    loader,
    device,
    condition,
):

    all_targets = []
    all_logits = []
    all_losses = []
    all_preds = []
    all_correct = []

    added_head_log = {
        i: []
        for i in range(model.depth)
    }

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

        # ----------------------------------------------------
        # Standard
        # ----------------------------------------------------

        if condition["type"] == "STANDARD":

            logits = model(
                images,
                budget=condition[
                    "budget"
                ],
            )

            infos = []

        # ----------------------------------------------------
        # Custom
        # ----------------------------------------------------

        else:

            logits, infos = (
                custom_model_forward(
                    model,
                    images,
                    add_mode=condition[
                        "add_mode"
                    ],
                    target_block=condition[
                        "target_block"
                    ],
                )
            )

        stats = stats_from_logits(
            logits,
            targets,
        )

        all_targets.append(
            targets.cpu()
        )

        all_logits.append(
            stats["logits"]
        )

        all_losses.append(
            stats["loss"]
        )

        all_preds.append(
            stats["pred"]
        )

        all_correct.append(
            stats["correct"]
        )

        # ----------------------------------------------------
        # Added head logging
        # ----------------------------------------------------

        for block_idx, info in infos:

            added_head_log[
                block_idx
            ].append(
                info[
                    "added_heads"
                ].cpu()
            )

        if (
            batch_idx % 10 == 0
            or
            batch_idx + 1
            == len(loader)
        ):

            print(
                f"{condition['name']} "
                f"[{batch_idx + 1}/"
                f"{len(loader)}]"
            )

    result = {
        "targets": torch.cat(
            all_targets
        ),

        "logits": torch.cat(
            all_logits
        ),

        "loss": torch.cat(
            all_losses
        ),

        "pred": torch.cat(
            all_preds
        ),

        "correct": torch.cat(
            all_correct
        ),

        "added_heads": {},
    }

    for block_idx in range(
        model.depth
    ):

        if added_head_log[
            block_idx
        ]:

            result[
                "added_heads"
            ][block_idx] = torch.cat(
                added_head_log[
                    block_idx
                ],
                dim=0,
            )

    return result


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap_ci(
    baseline,
    condition,
    repeats,
    seed,
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64,
    )

    condition = np.asarray(
        condition,
        dtype=np.float64,
    )

    delta = (
        condition
        -
        baseline
    )

    n = len(delta)

    rng = np.random.default_rng(seed)

    boot = np.empty(
        repeats,
        dtype=np.float64,
    )

    chunk = 100

    done = 0

    while done < repeats:

        r = min(
            chunk,
            repeats - done,
        )

        idx = rng.integers(
            0,
            n,
            size=(r, n),
        )

        boot[
            done:done+r
        ] = delta[
            idx
        ].mean(axis=1)

        done += r

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return (
        delta.mean(),
        lo,
        hi,
    )


# ============================================================
# McNemar
# ============================================================

def exact_mcnemar(
    baseline_correct,
    condition_correct,
):

    base = np.asarray(
        baseline_correct,
        dtype=bool,
    )

    cond = np.asarray(
        condition_correct,
        dtype=bool,
    )

    degraded = int(
        np.sum(
            base & ~cond
        )
    )

    rescued = int(
        np.sum(
            ~base & cond
        )
    )

    n = degraded + rescued

    if n == 0:
        return (
            degraded,
            rescued,
            1.0,
        )

    try:
        from scipy.stats import binomtest

        p = binomtest(
            min(
                degraded,
                rescued,
            ),
            n=n,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    except ImportError:

        k = min(
            degraded,
            rescued,
        )

        tail = sum(
            math.comb(n, i)
            for i in range(k + 1)
        ) / (2 ** n)

        p = min(
            1.0,
            2.0 * tail,
        )

    return (
        degraded,
        rescued,
        float(p),
    )


# ============================================================
# Summary
# ============================================================

def summarize_condition(
    name,
    result,
    baseline,
    bootstrap,
    seed,
):

    correct = result[
        "correct"
    ].numpy()

    loss = result[
        "loss"
    ].numpy()

    base_correct = baseline[
        "correct"
    ].numpy()

    base_loss = baseline[
        "loss"
    ].numpy()

    acc = (
        correct.mean()
        * 100.0
    )

    ce = loss.mean()

    (
        dacc,
        dacc_lo,
        dacc_hi,
    ) = paired_bootstrap_ci(
        base_correct.astype(
            np.float64
        ) * 100,
        correct.astype(
            np.float64
        ) * 100,
        bootstrap,
        seed,
    )

    (
        dce,
        dce_lo,
        dce_hi,
    ) = paired_bootstrap_ci(
        base_loss,
        loss,
        bootstrap,
        seed + 1000,
    )

    (
        degraded,
        rescued,
        mcnemar_p,
    ) = exact_mcnemar(
        base_correct,
        correct,
    )

    return {
        "condition":
            name,

        "ce":
            ce,

        "accuracy":
            acc,

        "delta_ce_vs_b6":
            dce,

        "delta_ce_ci_low":
            dce_lo,

        "delta_ce_ci_high":
            dce_hi,

        "delta_acc_vs_b6":
            dacc,

        "delta_acc_ci_low":
            dacc_lo,

        "delta_acc_ci_high":
            dacc_hi,

        "b6_correct_to_condition_wrong":
            degraded,

        "b6_wrong_to_condition_correct":
            rescued,

        "mcnemar_p":
            mcnemar_p,
    }


# ============================================================
# Head frequency
# ============================================================

def print_added_head_frequency(
    name,
    result,
    main_heads,
):

    if not result[
        "added_heads"
    ]:
        return []

    rows = []

    print()
    print(
        f"Added-head frequency: {name}"
    )

    for block_idx, added in (
        result[
            "added_heads"
        ].items()
    ):

        print(
            f"  Block {block_idx}"
        )

        flat = (
            added
            .reshape(-1)
            .numpy()
        )

        for h in range(main_heads):

            count = int(
                np.sum(flat == h)
            )

            total_samples = (
                added.shape[0]
            )

            # For BOTH, there are two added heads
            # per sample, so frequency here means:
            # "% samples where Hh is among added heads".
            if added.shape[1] == 1:

                pct = (
                    100.0
                    * count
                    / total_samples
                )

            else:

                pct = (
                    100.0
                    * np.mean(
                        (
                            added.numpy()
                            == h
                        ).any(axis=1)
                    )
                )

            print(
                f"    H{h}: "
                f"{pct:6.2f}%"
            )

            rows.append({
                "condition":
                    name,

                "block":
                    block_idx,

                "head":
                    h,

                "sample_frequency_pct":
                    pct,
            })

    return rows


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
        "B6 EXCLUDED-HEAD MARGINAL ANALYSIS"
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

    print()
    print(
        f"Mini heads : {model.mini_heads}"
    )

    print(
        f"Main heads : {model.main_heads}"
    )

    print(
        f"Depth      : {model.depth}"
    )

    # --------------------------------------------------------
    # Data
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

    # ========================================================
    # Conditions
    # ========================================================

    conditions = [
        {
            "name": "B6",
            "type": "STANDARD",
            "budget": 6,
        },

        {
            "name": "B8",
            "type": "STANDARD",
            "budget": 8,
        },

        {
            "name": "ALL_NEXT",
            "type": "CUSTOM",
            "add_mode": "NEXT",
            "target_block": None,
        },

        {
            "name": "ALL_LAST",
            "type": "CUSTOM",
            "add_mode": "LAST",
            "target_block": None,
        },

        {
            "name": "ALL_BOTH",
            "type": "CUSTOM",
            "add_mode": "BOTH",
            "target_block": None,
        },
    ]

    if not args.skip_isolated:

        for block_idx in range(
            model.depth
        ):

            for mode in [
                "NEXT",
                "LAST",
                "BOTH",
            ]:

                conditions.append({
                    "name":
                        f"BLOCK{block_idx}_{mode}",

                    "type":
                        "CUSTOM",

                    "add_mode":
                        mode,

                    "target_block":
                        block_idx,
                })

    # ========================================================
    # Evaluate
    # ========================================================

    results = {}

    for condition in conditions:

        print()
        print("=" * 78)
        print(
            f"Evaluating {condition['name']}"
        )
        print("=" * 78)

        results[
            condition["name"]
        ] = evaluate_condition(
            model,
            loader,
            device,
            condition,
        )

    # ========================================================
    # Sanity
    # ========================================================

    b6 = results["B6"]
    b8 = results["B8"]
    all_both = results["ALL_BOTH"]

    print()
    print("=" * 78)
    print("SANITY CHECK")
    print("=" * 78)

    b6_ce = (
        b6["loss"].mean().item()
    )

    b6_acc = (
        b6["correct"]
        .float()
        .mean()
        .item()
        * 100
    )

    b8_ce = (
        b8["loss"].mean().item()
    )

    b8_acc = (
        b8["correct"]
        .float()
        .mean()
        .item()
        * 100
    )

    print(
        f"B6 CE / Acc : "
        f"{b6_ce:.8f} / "
        f"{b6_acc:.4f}%"
    )

    print(
        f"B8 CE / Acc : "
        f"{b8_ce:.8f} / "
        f"{b8_acc:.4f}%"
    )

    max_logit_error = (
        all_both["logits"]
        - b8["logits"]
    ).abs().max().item()

    mean_logit_error = (
        all_both["logits"]
        - b8["logits"]
    ).abs().mean().item()

    same_prediction = (
        all_both["pred"]
        == b8["pred"]
    ).float().mean().item() * 100

    print()
    print(
        "ALL_BOTH vs standard B8"
    )

    print(
        f"Mean |logit error| : "
        f"{mean_logit_error:.10f}"
    )

    print(
        f"Max |logit error|  : "
        f"{max_logit_error:.10f}"
    )

    print(
        f"Prediction same    : "
        f"{same_prediction:.4f}%"
    )

    if max_logit_error > 1e-5:

        print()
        print(
            "[WARNING] ALL_BOTH should "
            "closely reproduce standard B8."
        )

    # ========================================================
    # Statistical summary
    # ========================================================

    rows = []

    print()
    print("=" * 78)
    print("RESULT SUMMARY")
    print("=" * 78)

    for idx, condition in enumerate(
        conditions
    ):

        name = condition["name"]

        if name == "B6":

            result = results[name]

            row = {
                "condition": "B6",
                "ce":
                    result[
                        "loss"
                    ].mean().item(),

                "accuracy":
                    result[
                        "correct"
                    ].float().mean().item()
                    * 100,

                "delta_ce_vs_b6":
                    0.0,

                "delta_ce_ci_low":
                    0.0,

                "delta_ce_ci_high":
                    0.0,

                "delta_acc_vs_b6":
                    0.0,

                "delta_acc_ci_low":
                    0.0,

                "delta_acc_ci_high":
                    0.0,

                "b6_correct_to_condition_wrong":
                    0,

                "b6_wrong_to_condition_correct":
                    0,

                "mcnemar_p":
                    1.0,
            }

        else:

            row = summarize_condition(
                name,
                results[name],
                b6,
                args.bootstrap,
                args.seed + idx,
            )

        rows.append(row)

        print()
        print(
            f"{name}"
        )

        print(
            f"  CE                       : "
            f"{row['ce']:.8f}"
        )

        print(
            f"  Accuracy                 : "
            f"{row['accuracy']:.4f}%"
        )

        print(
            f"  ΔCE vs B6               : "
            f"{row['delta_ce_vs_b6']:+.8f}"
        )

        print(
            f"  CE 95% CI               : "
            f"[{row['delta_ce_ci_low']:+.8f}, "
            f"{row['delta_ce_ci_high']:+.8f}]"
        )

        print(
            f"  ΔAcc vs B6              : "
            f"{row['delta_acc_vs_b6']:+.4f}%p"
        )

        print(
            f"  Acc 95% CI              : "
            f"[{row['delta_acc_ci_low']:+.4f}, "
            f"{row['delta_acc_ci_high']:+.4f}]%p"
        )

        print(
            f"  B6 correct -> wrong     : "
            f"{row['b6_correct_to_condition_wrong']}"
        )

        print(
            f"  B6 wrong -> correct     : "
            f"{row['b6_wrong_to_condition_correct']}"
        )

        print(
            f"  McNemar p               : "
            f"{row['mcnemar_p']:.6f}"
        )

    # ========================================================
    # Added-head frequencies
    # ========================================================

    freq_rows = []

    print()
    print("=" * 78)
    print("WHICH HEAD IS BEING ADDED?")
    print("=" * 78)

    for condition in conditions:

        name = condition["name"]

        freq_rows.extend(
            print_added_head_frequency(
                name,
                results[name],
                model.main_heads,
            )
        )

    # ========================================================
    # Save
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "marginal_head_summary.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        summary_path,
        index=False,
    )

    freq_path = (
        output_dir
        / "added_head_frequency.csv"
    )

    pd.DataFrame(
        freq_rows
    ).to_csv(
        freq_path,
        index=False,
    )

    # --------------------------------------------------------
    # Per sample major conditions
    # --------------------------------------------------------

    per_sample = pd.DataFrame({
        "heldout_position":
            np.arange(
                len(heldout_indices)
            ),

        "original_cifar_index":
            heldout_indices,

        "target":
            b6[
                "targets"
            ].numpy(),

        "b6_pred":
            results[
                "B6"
            ]["pred"].numpy(),

        "b8_pred":
            results[
                "B8"
            ]["pred"].numpy(),

        "next_pred":
            results[
                "ALL_NEXT"
            ]["pred"].numpy(),

        "last_pred":
            results[
                "ALL_LAST"
            ]["pred"].numpy(),

        "b6_loss":
            results[
                "B6"
            ]["loss"].numpy(),

        "b8_loss":
            results[
                "B8"
            ]["loss"].numpy(),

        "next_loss":
            results[
                "ALL_NEXT"
            ]["loss"].numpy(),

        "last_loss":
            results[
                "ALL_LAST"
            ]["loss"].numpy(),
    })

    sample_path = (
        output_dir
        / "marginal_head_per_sample.csv"
    )

    per_sample.to_csv(
        sample_path,
        index=False,
    )

    # ========================================================
    # Final interpretation
    # ========================================================

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)

    print(
        """
핵심 비교 1

B6 vs ALL_NEXT

→ B6가 제외한 head 중
  상대적으로 need score가 높은
  '다음 후보 head'를 추가했을 때
  성능이 좋아지는지 확인.


핵심 비교 2

B6 vs ALL_LAST

→ B6가 가장 필요 없다고 판단한
  마지막 head를 추가했을 때
  성능이 떨어지는지 확인.


가능한 결과 A

ALL_NEXT > B6
ALL_LAST < B6
B8 ≈ 또는 < B6

→ 마지막 excluded head가
  negative marginal utility를 가질 가능성.
→ budget이 커졌다고 모든 head를
  켜는 것이 최적이 아닐 수 있음.


가능한 결과 B

ALL_NEXT > B6
ALL_LAST > B6
B8 < B6

→ 개별 head 자체보다
  두 head 동시 활성화 / downstream interaction
  문제가 의심됨.


가능한 결과 C

ALL_NEXT ≈ B6
ALL_LAST ≈ B6
B8 ≈ B6

→ B6 이후 거의 saturation.
→ 기존 -0.22%p는
  통계 fluctuation일 가능성이 큼.


가능한 결과 D

특정 BLOCKk_LAST만
accuracy/CE를 악화시킴

→ B8 문제의 위치를
  특정 block까지 좁힐 수 있음.


주의

ALL_NEXT / ALL_LAST는
training에 사용한 공식 budget이 아니라
diagnostic B7-like counterfactual이다.

따라서 이것 자체를 최종 성능으로
주장하는 실험이 아니라,
B6 -> B8 변화 원인을 찾기 위한
causal diagnostic으로 해석해야 한다.
        """
    )

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)

    print(summary_path)
    print(freq_path)
    print(sample_path)


if __name__ == "__main__":
    main()