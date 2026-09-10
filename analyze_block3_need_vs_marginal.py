import argparse
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
            "block3_need_vs_marginal"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--bootstrap",
        type=int,
        default=5000,
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

def build_heldout_dataset(
    data_dir,
    seed,
):
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
        transforms.Normalize(
            mean,
            std,
        ),
    ])

    base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    if len(base) != 50000:
        raise RuntimeError(
            f"Expected 50000 samples, got {len(base)}"
        )

    g = torch.Generator().manual_seed(
        seed
    )

    perm = torch.randperm(
        50000,
        generator=g,
    ).tolist()

    # Same fair split:
    #
    # train   : 0~39999
    # val     : 40000~44999
    # heldout : 45000~49999

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

def load_checkpoint(
    path,
    device,
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

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


def build_model(
    ckpt,
    device,
):
    cfg = ckpt.get(
        "config",
        {},
    )

    model = BudgetedMiniMainViTV2(
        img_size=cfg.get(
            "img_size",
            32,
        ),

        patch_size=cfg.get(
            "patch_size",
            4,
        ),

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

        mlp_ratio=cfg.get(
            "mlp_ratio",
            4.0,
        ),

        drop_rate=cfg.get(
            "drop_rate",
            0.0,
        ),

        attn_drop_rate=cfg.get(
            "attn_drop_rate",
            0.0,
        ),

        drop_path_rate=cfg.get(
            "drop_path_rate",
            0.0,
        ),

        route_tau=cfg.get(
            "route_tau",
            1.0,
        ),

        bind_tau=cfg.get(
            "bind_tau",
            1.0,
        ),
    ).to(device)

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# Basic stats
# ============================================================

def logits_stats(
    logits,
    targets,
):
    losses = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    preds = logits.argmax(
        dim=1
    )

    correct = (
        preds == targets
    )

    return (
        losses,
        preds,
        correct,
    )


# ============================================================
# Finish one branch after Block 3 attention
# ============================================================

def finish_block_and_classifier(
    model,
    block,
    x_before_block,
    attn_out,
):
    """
    x_before_block:
        input to Block 3 before norm1

    attn_out:
        Mini base + Main output
    """

    x = (
        x_before_block
        + block.dp1(attn_out)
    )

    x = (
        x
        + block.dp2(
            block.mlp(
                block.norm2(x)
            )
        )
    )

    cls = model.norm(x)[:, 0]

    return model.head(cls)


# ============================================================
# Main counterfactual forward
# ============================================================

@torch.inference_mode()
def forward_block3_counterfactual(
    model,
    images,
):
    """
    Blocks 0,1,2:
        ordinary B6.

    Block 3:
        Build Mini / Utility / Binding / seeds ONCE
        under the same B6 state.

        Then branch into:

        BASE:
            original B6 active 6 heads

        NEXT:
            + higher-need excluded head

        LAST:
            + lower-need excluded head

    Therefore comparison is isolated to Block 3.
    """

    B = images.shape[0]

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Blocks 0~2: ordinary B6
    # --------------------------------------------------------

    for block_idx in range(
        model.depth - 1
    ):
        x = model.blocks[
            block_idx
        ](
            x,
            budget=6,
            patch_hw=model.patch_hw,
            force_dense_main=False,
            return_info=False,
        )

    # This is EXACT same Block 3 input
    # for BASE / NEXT / LAST.

    x_block3 = x

    block = model.blocks[-1]

    z = block.norm1(
        x_block3
    )

    attn = block.attn

    # --------------------------------------------------------
    # Mini
    # --------------------------------------------------------

    mini_contexts, mini_attn = (
        attn.mini(
            z,
            model.patch_hw,
        )
    )

    utility_logits = attn.utility(
        mini_contexts,
        mini_attn,
    )

    N = mini_contexts.shape[2]

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
    # Router:
    # Always compute ORIGINAL B6 route.
    # --------------------------------------------------------

    seeds, route_info = (
        attn.router(
            mini_contexts,
            utility_logits,
            budget=6,
            training_st=False,
        )
    )

    b6_mask = route_info[
        "active_main_mask"
    ]

    need_scores = route_info[
        "need_scores"
    ]

    # --------------------------------------------------------
    # Find B6-excluded 2 heads
    # --------------------------------------------------------

    excluded_mask = (
        ~b6_mask
    )

    counts = excluded_mask.sum(
        dim=1
    )

    if not torch.all(
        counts == 2
    ):
        raise RuntimeError(
            "B6 should exclude exactly 2 Main heads."
        )

    # Higher need-score excluded head
    high_scores = need_scores.masked_fill(
        ~excluded_mask,
        -float("inf"),
    )

    next_head = high_scores.argmax(
        dim=1
    )

    # Lower need-score excluded head
    low_scores = need_scores.masked_fill(
        ~excluded_mask,
        float("inf"),
    )

    last_head = low_scores.argmin(
        dim=1
    )

    next_score = need_scores.gather(
        1,
        next_head[:, None],
    ).squeeze(1)

    last_score = need_scores.gather(
        1,
        last_head[:, None],
    ).squeeze(1)

    # --------------------------------------------------------
    # Active masks
    # --------------------------------------------------------

    next_mask = b6_mask.clone()

    next_mask.scatter_(
        1,
        next_head[:, None],
        True,
    )

    last_mask = b6_mask.clone()

    last_mask.scatter_(
        1,
        last_head[:, None],
        True,
    )

    # --------------------------------------------------------
    # Main attention
    # --------------------------------------------------------

    b6_main = attn.main(
        z,
        seeds,
        b6_mask,
        b6_mask.to(z.dtype),
        6,
        False,
        False,
    )

    next_main = attn.main(
        z,
        seeds,
        next_mask,
        next_mask.to(z.dtype),
        7,
        False,
        False,
    )

    last_main = attn.main(
        z,
        seeds,
        last_mask,
        last_mask.to(z.dtype),
        7,
        False,
        False,
    )

    # --------------------------------------------------------
    # Mini base stays IDENTICAL
    # --------------------------------------------------------

    mini_component = (
        attn.mini_base_scale
        * mini_base
    )

    b6_attn_out = (
        mini_component
        + b6_main
    )

    next_attn_out = (
        mini_component
        + next_main
    )

    last_attn_out = (
        mini_component
        + last_main
    )

    # --------------------------------------------------------
    # Finish final block
    # --------------------------------------------------------

    b6_logits = finish_block_and_classifier(
        model,
        block,
        x_block3,
        b6_attn_out,
    )

    next_logits = finish_block_and_classifier(
        model,
        block,
        x_block3,
        next_attn_out,
    )

    last_logits = finish_block_and_classifier(
        model,
        block,
        x_block3,
        last_attn_out,
    )

    return {
        "b6_logits":
            b6_logits,

        "next_logits":
            next_logits,

        "last_logits":
            last_logits,

        "next_head":
            next_head,

        "last_head":
            last_head,

        "next_score":
            next_score,

        "last_score":
            last_score,

        "need_scores":
            need_scores,

        "b6_active_mask":
            b6_mask,
    }


# ============================================================
# Bootstrap mean CI
# ============================================================

def bootstrap_mean_ci(
    values,
    repeats=5000,
    seed=42,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

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
        ] = values[
            idx
        ].mean(axis=1)

        done += r

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return (
        values.mean(),
        lo,
        hi,
    )


# ============================================================
# Correlations
# ============================================================

def correlations(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    try:
        from scipy.stats import (
            pearsonr,
            spearmanr,
        )

        pearson = pearsonr(
            x,
            y,
        )

        spearman = spearmanr(
            x,
            y,
        )

        return {
            "pearson_r":
                float(pearson.statistic),

            "pearson_p":
                float(pearson.pvalue),

            "spearman_r":
                float(spearman.statistic),

            "spearman_p":
                float(spearman.pvalue),
        }

    except Exception:

        pearson_r = np.corrcoef(
            x,
            y,
        )[0, 1]

        return {
            "pearson_r":
                float(pearson_r),

            "pearson_p":
                np.nan,

            "spearman_r":
                np.nan,

            "spearman_p":
                np.nan,
        }


# ============================================================
# Exact binomial test
# ============================================================

def binomial_vs_random(
    successes,
    trials,
):
    if trials == 0:
        return np.nan

    try:
        from scipy.stats import binomtest

        return float(
            binomtest(
                successes,
                trials,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    except Exception:
        return np.nan


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):

    storage = {
        "target": [],

        "b6_loss": [],
        "b6_pred": [],
        "b6_correct": [],

        "next_loss": [],
        "next_pred": [],
        "next_correct": [],

        "last_loss": [],
        "last_pred": [],
        "last_correct": [],

        "next_head": [],
        "last_head": [],

        "next_score": [],
        "last_score": [],
    }

    sanity_done = False

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

        out = (
            forward_block3_counterfactual(
                model,
                images,
            )
        )

        # ----------------------------------------------------
        # Sanity:
        # custom B6 must reproduce standard B6.
        # Only first batch needed.
        # ----------------------------------------------------

        if not sanity_done:

            standard_b6 = model(
                images,
                budget=6,
            )

            mean_error = (
                standard_b6
                - out["b6_logits"]
            ).abs().mean().item()

            max_error = (
                standard_b6
                - out["b6_logits"]
            ).abs().max().item()

            prediction_same = (
                standard_b6.argmax(dim=1)
                ==
                out[
                    "b6_logits"
                ].argmax(dim=1)
            ).float().mean().item()

            print()
            print("=" * 78)
            print("FIRST-BATCH SANITY")
            print("=" * 78)

            print(
                f"Mean |logit error|        : "
                f"{mean_error:.10f}"
            )

            print(
                f"Max |logit error|         : "
                f"{max_error:.10f}"
            )

            print(
                f"Prediction same           : "
                f"{prediction_same*100:.4f}%"
            )

            if max_error > 1e-5:
                print()
                print(
                    "[WARNING] Custom B6 does not "
                    "closely reproduce standard B6."
                )

            sanity_done = True

        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------

        (
            b6_loss,
            b6_pred,
            b6_correct,
        ) = logits_stats(
            out["b6_logits"],
            targets,
        )

        (
            next_loss,
            next_pred,
            next_correct,
        ) = logits_stats(
            out["next_logits"],
            targets,
        )

        (
            last_loss,
            last_pred,
            last_correct,
        ) = logits_stats(
            out["last_logits"],
            targets,
        )

        batch_values = {
            "target":
                targets,

            "b6_loss":
                b6_loss,

            "b6_pred":
                b6_pred,

            "b6_correct":
                b6_correct,

            "next_loss":
                next_loss,

            "next_pred":
                next_pred,

            "next_correct":
                next_correct,

            "last_loss":
                last_loss,

            "last_pred":
                last_pred,

            "last_correct":
                last_correct,

            "next_head":
                out["next_head"],

            "last_head":
                out["last_head"],

            "next_score":
                out["next_score"],

            "last_score":
                out["last_score"],
        }

        for key, value in (
            batch_values.items()
        ):
            storage[key].append(
                value.detach().cpu()
            )

        if (
            batch_idx % 10 == 0
            or
            batch_idx + 1
            == len(loader)
        ):
            print(
                f"[{batch_idx + 1}/"
                f"{len(loader)}]"
            )

    for key in storage:
        storage[key] = torch.cat(
            storage[key],
            dim=0,
        ).numpy()

    return storage


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
        "BLOCK 3 NEED SCORE vs "
        "ACTUAL MARGINAL UTILITY"
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

    if model.depth != 4:
        raise RuntimeError(
            "This diagnostic currently expects depth=4."
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
        drop_last=False,
    )

    print(
        f"Heldout    : {len(heldout)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    data = evaluate(
        model,
        loader,
        device,
    )

    # ========================================================
    # Definitions
    # ========================================================

    b6_loss = data[
        "b6_loss"
    ]

    next_loss = data[
        "next_loss"
    ]

    last_loss = data[
        "last_loss"
    ]

    b6_correct = data[
        "b6_correct"
    ].astype(bool)

    next_correct = data[
        "next_correct"
    ].astype(bool)

    last_correct = data[
        "last_correct"
    ].astype(bool)

    next_score = data[
        "next_score"
    ]

    last_score = data[
        "last_score"
    ]

    # Positive benefit = adding head improved CE.
    next_benefit = (
        b6_loss
        - next_loss
    )

    last_benefit = (
        b6_loss
        - last_loss
    )

    # Need-score gap:
    # positive by construction.
    score_gap = (
        next_score
        - last_score
    )

    # Positive means NEXT head
    # truly has better marginal utility.
    benefit_gap = (
        next_benefit
        - last_benefit
    )

    # Equivalent:
    # benefit_gap = last_loss - next_loss

    # ========================================================
    # Headline
    # ========================================================

    print()
    print("=" * 78)
    print("HEADLINE PERFORMANCE")
    print("=" * 78)

    def print_condition(
        name,
        loss,
        correct,
    ):
        print(
            f"{name:<10} "
            f"CE={loss.mean():.8f}  "
            f"Acc={correct.mean()*100:.4f}%"
        )

    print_condition(
        "B6",
        b6_loss,
        b6_correct,
    )

    print_condition(
        "NEXT",
        next_loss,
        next_correct,
    )

    print_condition(
        "LAST",
        last_loss,
        last_correct,
    )

    # ========================================================
    # Actual marginal utility
    # ========================================================

    print()
    print("=" * 78)
    print("ACTUAL MARGINAL UTILITY")
    print("=" * 78)

    (
        next_mean,
        next_lo,
        next_hi,
    ) = bootstrap_mean_ci(
        next_benefit,
        args.bootstrap,
        args.seed,
    )

    (
        last_mean,
        last_lo,
        last_hi,
    ) = bootstrap_mean_ci(
        last_benefit,
        args.bootstrap,
        args.seed + 1,
    )

    print(
        "Definition: "
        "benefit = CE(B6) - CE(B6 + head)"
    )

    print(
        "Positive = useful, Negative = harmful"
    )

    print()
    print(
        f"NEXT benefit mean          : "
        f"{next_mean:+.8f}"
    )

    print(
        f"NEXT 95% CI               : "
        f"[{next_lo:+.8f}, "
        f"{next_hi:+.8f}]"
    )

    print(
        f"NEXT beneficial samples    : "
        f"{np.mean(next_benefit > 0)*100:.2f}%"
    )

    print()
    print(
        f"LAST benefit mean          : "
        f"{last_mean:+.8f}"
    )

    print(
        f"LAST 95% CI               : "
        f"[{last_lo:+.8f}, "
        f"{last_hi:+.8f}]"
    )

    print(
        f"LAST beneficial samples    : "
        f"{np.mean(last_benefit > 0)*100:.2f}%"
    )

    # ========================================================
    # Core test:
    # Does higher need score choose the better head?
    # ========================================================

    eps = 1e-8

    valid = (
        np.abs(
            next_loss
            - last_loss
        )
        > eps
    )

    next_actually_better = (
        next_loss
        < last_loss
    )

    n_valid = int(
        valid.sum()
    )

    n_correct_rank = int(
        (
            next_actually_better
            & valid
        ).sum()
    )

    rank_accuracy = (
        n_correct_rank
        / n_valid
        if n_valid > 0
        else np.nan
    )

    rank_p = binomial_vs_random(
        n_correct_rank,
        n_valid,
    )

    (
        rank_mean,
        rank_lo,
        rank_hi,
    ) = bootstrap_mean_ci(
        next_actually_better[
            valid
        ].astype(
            np.float64
        ) * 100.0,
        args.bootstrap,
        args.seed + 2,
    )

    print()
    print("=" * 78)
    print("PRIMARY TEST — NEED-SCORE RANKING QUALITY")
    print("=" * 78)

    print(
        "Router prediction:"
    )

    print(
        "  NEXT score > LAST score"
    )

    print()
    print(
        "Question:"
    )

    print(
        "  Is NEXT actually lower-CE "
        "than LAST?"
    )

    print()
    print(
        f"Valid comparisons          : "
        f"{n_valid}"
    )

    print(
        f"Higher-score head wins     : "
        f"{n_correct_rank}"
    )

    print(
        f"Ranking accuracy           : "
        f"{rank_accuracy*100:.2f}%"
    )

    print(
        f"Bootstrap 95% CI           : "
        f"[{rank_lo:.2f}, "
        f"{rank_hi:.2f}]%"
    )

    print(
        f"Random baseline            : "
        f"50.00%"
    )

    print(
        f"Binomial p vs 50%          : "
        f"{rank_p:.8f}"
    )

    # ========================================================
    # Gap correlation
    # ========================================================

    gap_corr = correlations(
        score_gap,
        benefit_gap,
    )

    print()
    print("=" * 78)
    print("PAIRWISE GAP CORRELATION")
    print("=" * 78)

    print(
        "x = need_score(NEXT) "
        "- need_score(LAST)"
    )

    print(
        "y = benefit(NEXT) "
        "- benefit(LAST)"
    )

    print()
    print(
        f"Pearson r                  : "
        f"{gap_corr['pearson_r']:+.6f}"
    )

    print(
        f"Pearson p                  : "
        f"{gap_corr['pearson_p']:.8f}"
    )

    print(
        f"Spearman rho               : "
        f"{gap_corr['spearman_r']:+.6f}"
    )

    print(
        f"Spearman p                 : "
        f"{gap_corr['spearman_p']:.8f}"
    )

    # ========================================================
    # Supplementary pooled correlation
    # ========================================================

    pooled_score = np.concatenate([
        next_score,
        last_score,
    ])

    pooled_benefit = np.concatenate([
        next_benefit,
        last_benefit,
    ])

    pooled_corr = correlations(
        pooled_score,
        pooled_benefit,
    )

    print()
    print("=" * 78)
    print("POOLED SCORE ↔ BENEFIT CORRELATION")
    print("=" * 78)

    print(
        "Supplementary only: "
        "two observations per sample."
    )

    print()
    print(
        f"Pearson r                  : "
        f"{pooled_corr['pearson_r']:+.6f}"
    )

    print(
        f"Spearman rho               : "
        f"{pooled_corr['spearman_r']:+.6f}"
    )

    # ========================================================
    # Does adding a head help at all?
    # ========================================================

    next_better_than_base = (
        next_loss
        < b6_loss
    )

    last_better_than_base = (
        last_loss
        < b6_loss
    )

    both_help = (
        next_better_than_base
        & last_better_than_base
    )

    both_hurt = (
        ~next_better_than_base
        & ~last_better_than_base
    )

    only_next = (
        next_better_than_base
        & ~last_better_than_base
    )

    only_last = (
        ~next_better_than_base
        & last_better_than_base
    )

    print()
    print("=" * 78)
    print("SHOULD WE ADD A 7TH HEAD AT ALL?")
    print("=" * 78)

    print(
        f"Both additions improve CE : "
        f"{both_help.mean()*100:.2f}%"
    )

    print(
        f"Only NEXT improves         : "
        f"{only_next.mean()*100:.2f}%"
    )

    print(
        f"Only LAST improves         : "
        f"{only_last.mean()*100:.2f}%"
    )

    print(
        f"Both additions worsen CE  : "
        f"{both_hurt.mean()*100:.2f}%"
    )

    print(
        f"At least one helps         : "
        f"{(~both_hurt).mean()*100:.2f}%"
    )

    # ========================================================
    # Label-conditioned diagnostic oracles
    # ========================================================

    candidate_losses = np.stack(
        [
            next_loss,
            last_loss,
        ],
        axis=1,
    )

    candidate_correct = np.stack(
        [
            next_correct,
            last_correct,
        ],
        axis=1,
    )

    oracle_add_idx = np.argmin(
        candidate_losses,
        axis=1,
    )

    row_idx = np.arange(
        len(b6_loss)
    )

    oracle_add_loss = (
        candidate_losses[
            row_idx,
            oracle_add_idx,
        ]
    )

    oracle_add_correct = (
        candidate_correct[
            row_idx,
            oracle_add_idx,
        ]
    )

    # Also allow "do not add anything"
    all_choices_loss = np.stack(
        [
            b6_loss,
            next_loss,
            last_loss,
        ],
        axis=1,
    )

    all_choices_correct = np.stack(
        [
            b6_correct,
            next_correct,
            last_correct,
        ],
        axis=1,
    )

    oracle_stop_idx = np.argmin(
        all_choices_loss,
        axis=1,
    )

    oracle_stop_loss = (
        all_choices_loss[
            row_idx,
            oracle_stop_idx,
        ]
    )

    oracle_stop_correct = (
        all_choices_correct[
            row_idx,
            oracle_stop_idx,
        ]
    )

    print()
    print("=" * 78)
    print("DIAGNOSTIC ORACLE UPPER BOUNDS")
    print("=" * 78)

    print(
        "WARNING: uses ground-truth CE; "
        "not deployable."
    )

    print()
    print_condition(
        "B6",
        b6_loss,
        b6_correct,
    )

    print_condition(
        "NEXT",
        next_loss,
        next_correct,
    )

    print_condition(
        "LAST",
        last_loss,
        last_correct,
    )

    print_condition(
        "ORACLE+1",
        oracle_add_loss,
        oracle_add_correct,
    )

    print_condition(
        "ORACLESTOP",
        oracle_stop_loss,
        oracle_stop_correct,
    )

    # What does the stop/add oracle prefer?
    option_names = [
        "B6_STOP",
        "NEXT",
        "LAST",
    ]

    print()
    print("Best CE option per sample:")

    for i, name in enumerate(
        option_names
    ):
        count = int(
            np.sum(
                oracle_stop_idx == i
            )
        )

        print(
            f"  {name:<8}: "
            f"{count:4d} "
            f"({100*count/len(b6_loss):6.2f}%)"
        )

    # ========================================================
    # Accuracy flips
    # ========================================================

    print()
    print("=" * 78)
    print("ACCURACY FLIPS vs B6")
    print("=" * 78)

    for name, correct in [
        (
            "NEXT",
            next_correct,
        ),
        (
            "LAST",
            last_correct,
        ),
    ]:

        degraded = np.sum(
            b6_correct
            & ~correct
        )

        rescued = np.sum(
            ~b6_correct
            & correct
        )

        print()
        print(name)

        print(
            f"  B6 correct -> wrong     : "
            f"{degraded}"
        )

        print(
            f"  B6 wrong -> correct     : "
            f"{rescued}"
        )

        print(
            f"  Net correct change      : "
            f"{int(rescued)-int(degraded):+d}"
        )

    # ========================================================
    # Per-head actual contribution
    # ========================================================

    candidate_head = np.stack(
        [
            data["next_head"],
            data["last_head"],
        ],
        axis=1,
    )

    candidate_score = np.stack(
        [
            next_score,
            last_score,
        ],
        axis=1,
    )

    candidate_benefit = np.stack(
        [
            next_benefit,
            last_benefit,
        ],
        axis=1,
    )

    candidate_loss = np.stack(
        [
            next_loss,
            last_loss,
        ],
        axis=1,
    )

    candidate_correct_matrix = (
        np.stack(
            [
                next_correct,
                last_correct,
            ],
            axis=1,
        )
    )

    base_correct_matrix = np.repeat(
        b6_correct[:, None],
        2,
        axis=1,
    )

    role = np.empty(
        candidate_head.shape,
        dtype=object,
    )

    role[:, 0] = "NEXT"
    role[:, 1] = "LAST"

    head_rows = []

    print()
    print("=" * 78)
    print("PER-HEAD MARGINAL UTILITY")
    print("=" * 78)

    for h in range(
        model.main_heads
    ):

        mask = (
            candidate_head == h
        )

        n = int(
            mask.sum()
        )

        if n == 0:
            continue

        scores_h = (
            candidate_score[
                mask
            ]
        )

        benefit_h = (
            candidate_benefit[
                mask
            ]
        )

        correct_h = (
            candidate_correct_matrix[
                mask
            ]
        )

        base_correct_h = (
            base_correct_matrix[
                mask
            ]
        )

        role_h = role[
            mask
        ]

        corr_h = correlations(
            scores_h,
            benefit_h,
        ) if n >= 3 else {
            "spearman_r": np.nan,
            "pearson_r": np.nan,
        }

        acc_delta = (
            correct_h.astype(
                np.float64
            ).mean()
            -
            base_correct_h.astype(
                np.float64
            ).mean()
        ) * 100.0

        row = {
            "head":
                h,

            "excluded_count":
                n,

            "excluded_sample_pct":
                n / len(b6_loss)
                * 100.0,

            "next_role_pct":
                np.mean(
                    role_h == "NEXT"
                ) * 100.0,

            "mean_need_score":
                scores_h.mean(),

            "mean_benefit":
                benefit_h.mean(),

            "median_benefit":
                np.median(
                    benefit_h
                ),

            "beneficial_pct":
                np.mean(
                    benefit_h > 0
                ) * 100.0,

            "accuracy_delta_pct":
                acc_delta,

            "score_benefit_pearson":
                corr_h[
                    "pearson_r"
                ],

            "score_benefit_spearman":
                corr_h[
                    "spearman_r"
                ],
        }

        head_rows.append(
            row
        )

        print()
        print(
            f"H{h}"
        )

        print(
            f"  Excluded observations    : "
            f"{n}"
        )

        print(
            f"  NEXT role                : "
            f"{row['next_role_pct']:.2f}%"
        )

        print(
            f"  Mean need score          : "
            f"{row['mean_need_score']:+.6f}"
        )

        print(
            f"  Mean actual benefit      : "
            f"{row['mean_benefit']:+.6f}"
        )

        print(
            f"  Median actual benefit    : "
            f"{row['median_benefit']:+.6f}"
        )

        print(
            f"  Benefit > 0              : "
            f"{row['beneficial_pct']:.2f}%"
        )

        print(
            f"  Accuracy Δ when added    : "
            f"{row['accuracy_delta_pct']:+.4f}%p"
        )

    # ========================================================
    # Save per-sample
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame({
        "heldout_position":
            np.arange(
                len(b6_loss)
            ),

        "original_cifar_index":
            heldout_indices,

        "target":
            data["target"],

        "b6_loss":
            b6_loss,

        "b6_correct":
            b6_correct.astype(int),

        "next_head":
            data["next_head"],

        "next_need_score":
            next_score,

        "next_loss":
            next_loss,

        "next_benefit":
            next_benefit,

        "next_correct":
            next_correct.astype(int),

        "last_head":
            data["last_head"],

        "last_need_score":
            last_score,

        "last_loss":
            last_loss,

        "last_benefit":
            last_benefit,

        "last_correct":
            last_correct.astype(int),

        "need_score_gap":
            score_gap,

        "benefit_gap_next_minus_last":
            benefit_gap,

        "higher_need_head_is_better":
            next_actually_better.astype(
                int
            ),

        "both_help":
            both_help.astype(int),

        "only_next_helps":
            only_next.astype(int),

        "only_last_helps":
            only_last.astype(int),

        "both_hurt":
            both_hurt.astype(int),

        "oracle_add_choice":
            np.where(
                oracle_add_idx == 0,
                "NEXT",
                "LAST",
            ),

        "oracle_stop_choice":
            np.array(
                option_names,
                dtype=object,
            )[
                oracle_stop_idx
            ],
    })

    sample_path = (
        output_dir
        /
        "block3_need_vs_marginal_per_sample.csv"
    )

    df.to_csv(
        sample_path,
        index=False,
    )

    # ========================================================
    # Save per-head
    # ========================================================

    head_path = (
        output_dir
        /
        "block3_need_vs_marginal_per_head.csv"
    )

    pd.DataFrame(
        head_rows
    ).to_csv(
        head_path,
        index=False,
    )

    # ========================================================
    # Save headline summary
    # ========================================================

    summary = {
        "n":
            len(b6_loss),

        "b6_ce":
            b6_loss.mean(),

        "b6_acc":
            b6_correct.mean()
            * 100,

        "next_ce":
            next_loss.mean(),

        "next_acc":
            next_correct.mean()
            * 100,

        "last_ce":
            last_loss.mean(),

        "last_acc":
            last_correct.mean()
            * 100,

        "next_mean_benefit":
            next_benefit.mean(),

        "last_mean_benefit":
            last_benefit.mean(),

        "need_ranking_accuracy_pct":
            rank_accuracy * 100,

        "need_ranking_ci_low":
            rank_lo,

        "need_ranking_ci_high":
            rank_hi,

        "need_ranking_p_vs_50":
            rank_p,

        "score_gap_benefit_gap_pearson":
            gap_corr[
                "pearson_r"
            ],

        "score_gap_benefit_gap_spearman":
            gap_corr[
                "spearman_r"
            ],

        "both_help_pct":
            both_help.mean()
            * 100,

        "only_next_help_pct":
            only_next.mean()
            * 100,

        "only_last_help_pct":
            only_last.mean()
            * 100,

        "both_hurt_pct":
            both_hurt.mean()
            * 100,

        "oracle_add_ce":
            oracle_add_loss.mean(),

        "oracle_add_acc":
            oracle_add_correct.mean()
            * 100,

        "oracle_stop_ce":
            oracle_stop_loss.mean(),

        "oracle_stop_acc":
            oracle_stop_correct.mean()
            * 100,

        "oracle_prefers_stop_pct":
            np.mean(
                oracle_stop_idx == 0
            ) * 100,
    }

    summary_path = (
        output_dir
        /
        "block3_need_vs_marginal_summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Final guide
    # ========================================================

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)

    print(
        """
가장 중요한 숫자는:

1) Ranking accuracy

   need_score가 더 높은 excluded head가
   실제로 더 낮은 CE를 만드는 비율.

   > 50%이고 유의함
      → need_score가 actual marginal utility와 정렬.

   ≈ 50%
      → need_score가 어떤 excluded head를
        추가해야 좋은지 거의 못 맞힘.

   < 50%이고 유의함
      → ranking이 실제 marginal utility와
        반대로 정렬될 가능성.


2) score_gap vs benefit_gap correlation

   need-score 차이가 클수록
   실제 performance 차이도 같은 방향으로 커지는가?

   positive correlation
      → router confidence가 의미 있음.

   ~0
      → score 차이가 실제 utility 차이를 설명하지 못함.


3) Both hurt %

   B6에서 제외된 두 head를
   어느 쪽을 추가해도 CE가 악화되는 sample 비율.

   높다면:
      budget이 더 있다고 무조건 head를
      추가하는 것이 최적이 아닐 수 있음.


4) ORACLE+1

   두 excluded heads 중 실제 더 좋은 것을
   label을 보고 선택했을 때의 upper bound.

   NEXT보다 크게 좋다면:
      head 선택 자체에 개선 가능성이 있음.


5) ORACLESTOP

   B6에서 멈추는 선택까지 허용한 diagnostic oracle.

   B6_STOP 비율이 높다면:
      '몇 개의 head를 사용할지' 자체도
      input-dependent decision일 필요가 있다는 신호.


중요:

ORACLE+1 / ORACLESTOP은
ground-truth label을 사용한 diagnostic upper bound이며
실제 inference 성능으로 주장하면 안 됨.
        """
    )

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)

    print(sample_path)
    print(head_path)
    print(summary_path)


if __name__ == "__main__":
    main()