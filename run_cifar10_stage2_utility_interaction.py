import argparse
import itertools
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.utility_interaction_predictor import (
    UtilityInteractionPredictor,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--teacher-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_counterfactual_teacher_cache.pt"
        ),
    )

    p.add_argument(
        "--feature-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_subset_feature_cache.pt"
        ),
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_utility_interaction_predictor.pt"
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
        "--epochs",
        type=int,
        default=30,
    )

    p.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    # ---------------------------------------------------------
    # Loss weights
    # ---------------------------------------------------------

    # Direct pair quality
    p.add_argument(
        "--regret-weight",
        type=float,
        default=1.0,
    )

    # Pair-distribution supervision
    p.add_argument(
        "--pair-kl-weight",
        type=float,
        default=0.5,
    )

    # Preserve meaningful individual Mini utility
    p.add_argument(
        "--utility-kl-weight",
        type=float,
        default=0.5,
    )

    # Prevent the interaction term from swallowing all utility semantics
    p.add_argument(
        "--interaction-l2-weight",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--pair-temperature",
        type=float,
        default=0.5,
    )

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class CachedDataset(Dataset):
    def __init__(
        self,
        features,
        subset_losses,
        teacher_target,
    ):
        if (
            features.shape[0]
            !=
            subset_losses.shape[0]
        ):
            raise ValueError(
                "feature/subset-loss sample mismatch"
            )

        if (
            features.shape[0]
            !=
            teacher_target.shape[0]
        ):
            raise ValueError(
                "feature/utility-target sample mismatch"
            )

        self.features = (
            features.float()
        )

        self.subset_losses = (
            subset_losses.float()
        )

        self.teacher_target = (
            teacher_target.float()
        )

    def __len__(self):
        return (
            self.features.shape[0]
        )

    def __getitem__(
        self,
        idx,
    ):
        return (
            self.features[idx],
            self.subset_losses[idx],
            self.teacher_target[idx],
        )


# ============================================================
# Forward
# ============================================================

def forward_all_blocks(
    predictors,
    features,
    return_info=False,
):
    pair_scores = []
    utility_logits = []
    utility_pair_scores = []
    interactions = []

    depth = (
        features.shape[1]
    )

    for block_idx in range(
        depth
    ):
        scores, info = (
            predictors[
                block_idx
            ](
                features[
                    :,
                    block_idx,
                    :,
                    :,
                ],
                return_info=True,
            )
        )

        pair_scores.append(
            scores
        )

        utility_logits.append(
            info[
                "utility_logits"
            ]
        )

        utility_pair_scores.append(
            info[
                "utility_pair_scores"
            ]
        )

        interactions.append(
            info[
                "interaction_scores"
            ]
        )

    pair_scores = torch.stack(
        pair_scores,
        dim=1,
    )

    if not return_info:
        return pair_scores

    return pair_scores, {
        "utility_logits":
            torch.stack(
                utility_logits,
                dim=1,
            ),

        "utility_pair_scores":
            torch.stack(
                utility_pair_scores,
                dim=1,
            ),

        "interaction_scores":
            torch.stack(
                interactions,
                dim=1,
            ),
    }


# ============================================================
# Loss
# ============================================================

def compute_loss(
    pair_scores,
    utility_logits,
    interaction_scores,
    subset_losses,
    teacher_target,
    args,
    eps=1e-8,
):
    # ---------------------------------------------------------
    # Pair target
    # ---------------------------------------------------------

    min_loss = subset_losses.min(
        dim=-1,
        keepdim=True,
    ).values

    max_loss = subset_losses.max(
        dim=-1,
        keepdim=True,
    ).values

    spread = (
        max_loss
        -
        min_loss
    ).clamp_min(
        eps
    )

    normalized_cost = (
        subset_losses
        -
        min_loss
    ) / spread

    pair_probs = torch.softmax(
        pair_scores,
        dim=-1,
    )

    expected_regret = (
        pair_probs
        *
        normalized_cost
    ).sum(
        dim=-1
    ).mean()

    pair_target = torch.softmax(
        -normalized_cost
        /
        args.pair_temperature,
        dim=-1,
    )

    pair_kl = F.kl_div(
        F.log_softmax(
            pair_scores,
            dim=-1,
        ),
        pair_target,
        reduction="batchmean",
    )

    # ---------------------------------------------------------
    # Individual Utility target
    #
    # Uses the existing counterfactual per-head teacher target.
    # This is what preserves the original "each Mini has utility"
    # semantics.
    # ---------------------------------------------------------

    utility_kl = F.kl_div(
        F.log_softmax(
            utility_logits,
            dim=-1,
        ),
        teacher_target,
        reduction="batchmean",
    )

    # ---------------------------------------------------------
    # Interaction regularization
    # ---------------------------------------------------------

    interaction_l2 = (
        interaction_scores
        .pow(2)
        .mean()
    )

    total = (
        args.regret_weight
        *
        expected_regret
        +
        args.pair_kl_weight
        *
        pair_kl
        +
        args.utility_kl_weight
        *
        utility_kl
        +
        args.interaction_l2_weight
        *
        interaction_l2
    )

    return total, {
        "expected_regret":
            expected_regret.detach(),

        "pair_kl":
            pair_kl.detach(),

        "utility_kl":
            utility_kl.detach(),

        "interaction_l2":
            interaction_l2.detach(),
    }


# ============================================================
# Metrics
# ============================================================

def head_overlap(
    pred_pair,
    oracle_pair,
):
    match = (
        pred_pair[
            ...,
            :,
            None,
        ]
        ==
        oracle_pair[
            ...,
            None,
            :,
        ]
    )

    return (
        match
        .any(
            dim=-1
        )
        .float()
        .mean(
            dim=-1
        )
    )


def utility_topk_combo_index(
    utility_logits,
    combo_table,
):
    top2 = torch.topk(
        utility_logits,
        k=2,
        dim=-1,
    ).indices

    top2 = top2.sort(
        dim=-1
    ).values

    combo_table = combo_table.sort(
        dim=-1
    ).values

    equality = (
        top2[
            :,
            :,
            None,
            :,
        ]
        ==
        combo_table[
            None,
            None,
            :,
            :,
        ]
    ).all(
        dim=-1
    )

    return (
        equality
        .float()
        .argmax(
            dim=-1
        )
    )


@torch.no_grad()
def evaluate(
    predictors,
    loader,
    combo_table,
    device,
):
    for predictor in predictors:
        predictor.eval()

    combo_device = (
        combo_table.to(
            device
        )
    )

    final_exact_all = []
    final_overlap_all = []
    final_regret_all = []

    utility_exact_all = []
    utility_overlap_all = []
    utility_regret_all = []

    utility_teacher_top1_all = []
    utility_teacher_pair_all = []

    pred_idx_all = []

    interaction_abs_all = []

    for (
        features,
        subset_losses,
        teacher_target,
    ) in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        subset_losses = subset_losses.to(
            device,
            non_blocking=True,
        )

        teacher_target = teacher_target.to(
            device,
            non_blocking=True,
        )

        (
            pair_scores,
            info,
        ) = forward_all_blocks(
            predictors,
            features,
            return_info=True,
        )

        utility_logits = (
            info[
                "utility_logits"
            ]
        )

        interactions = (
            info[
                "interaction_scores"
            ]
        )

        # -----------------------------------------------------
        # Final corrected selection
        # -----------------------------------------------------

        pred_idx = (
            pair_scores.argmax(
                dim=-1
            )
        )

        oracle_idx = (
            subset_losses.argmin(
                dim=-1
            )
        )

        final_exact = (
            pred_idx
            ==
            oracle_idx
        ).float()

        pred_pair = (
            combo_device[
                pred_idx
            ]
        )

        oracle_pair = (
            combo_device[
                oracle_idx
            ]
        )

        final_overlap = head_overlap(
            pred_pair,
            oracle_pair,
        )

        final_selected_loss = (
            subset_losses.gather(
                dim=-1,
                index=pred_idx[
                    ...,
                    None,
                ],
            )
            .squeeze(-1)
        )

        oracle_loss = (
            subset_losses.min(
                dim=-1
            ).values
        )

        final_regret = (
            final_selected_loss
            -
            oracle_loss
        )

        # -----------------------------------------------------
        # Utility-only selection
        #
        # This is an internal ablation:
        # How well would individual utility work WITHOUT
        # pair interaction correction?
        # -----------------------------------------------------

        utility_idx = (
            utility_topk_combo_index(
                utility_logits,
                combo_device,
            )
        )

        utility_exact = (
            utility_idx
            ==
            oracle_idx
        ).float()

        utility_pair = (
            combo_device[
                utility_idx
            ]
        )

        utility_overlap = (
            head_overlap(
                utility_pair,
                oracle_pair,
            )
        )

        utility_selected_loss = (
            subset_losses.gather(
                dim=-1,
                index=utility_idx[
                    ...,
                    None,
                ],
            )
            .squeeze(-1)
        )

        utility_regret = (
            utility_selected_loss
            -
            oracle_loss
        )

        # -----------------------------------------------------
        # Utility teacher agreement
        # -----------------------------------------------------

        pred_top1 = (
            utility_logits.argmax(
                dim=-1
            )
        )

        teacher_top1 = (
            teacher_target.argmax(
                dim=-1
            )
        )

        utility_teacher_top1 = (
            pred_top1
            ==
            teacher_top1
        ).float()

        pred_top2 = torch.topk(
            utility_logits,
            k=2,
            dim=-1,
        ).indices.sort(
            dim=-1
        ).values

        teacher_top2 = torch.topk(
            teacher_target,
            k=2,
            dim=-1,
        ).indices.sort(
            dim=-1
        ).values

        utility_teacher_pair = (
            pred_top2
            ==
            teacher_top2
        ).all(
            dim=-1
        ).float()

        # -----------------------------------------------------

        final_exact_all.append(
            final_exact.cpu()
        )

        final_overlap_all.append(
            final_overlap.cpu()
        )

        final_regret_all.append(
            final_regret.cpu()
        )

        utility_exact_all.append(
            utility_exact.cpu()
        )

        utility_overlap_all.append(
            utility_overlap.cpu()
        )

        utility_regret_all.append(
            utility_regret.cpu()
        )

        utility_teacher_top1_all.append(
            utility_teacher_top1.cpu()
        )

        utility_teacher_pair_all.append(
            utility_teacher_pair.cpu()
        )

        pred_idx_all.append(
            pred_idx.cpu()
        )

        interaction_abs_all.append(
            interactions.abs().mean().cpu()
        )

    def cat_mean(items):
        return (
            torch.cat(
                items,
                dim=0,
            )
            .mean()
            .item()
        )

    final_regret = torch.cat(
        final_regret_all,
        dim=0,
    )

    utility_regret = torch.cat(
        utility_regret_all,
        dim=0,
    )

    return {
        "final_exact":
            cat_mean(
                final_exact_all
            ),

        "final_overlap":
            cat_mean(
                final_overlap_all
            ),

        "final_mean_regret":
            final_regret.mean().item(),

        "final_median_regret":
            final_regret.median().item(),

        "utility_exact":
            cat_mean(
                utility_exact_all
            ),

        "utility_overlap":
            cat_mean(
                utility_overlap_all
            ),

        "utility_mean_regret":
            utility_regret.mean().item(),

        "utility_teacher_top1":
            cat_mean(
                utility_teacher_top1_all
            ),

        "utility_teacher_pair":
            cat_mean(
                utility_teacher_pair_all
            ),

        "mean_abs_interaction":
            torch.stack(
                interaction_abs_all
            ).mean().item(),

        "pred_idx":
            torch.cat(
                pred_idx_all,
                dim=0,
            ),
    }


def evaluate_static(
    train_losses,
    val_losses,
    combo_table,
):
    static_idx = (
        train_losses.mean(
            dim=0
        )
        .argmin(
            dim=-1
        )
    )

    B = (
        val_losses.shape[0]
    )

    static_batch = (
        static_idx[
            None,
            :,
        ]
        .expand(
            B,
            -1,
        )
    )

    oracle_idx = (
        val_losses.argmin(
            dim=-1
        )
    )

    exact = (
        static_batch
        ==
        oracle_idx
    ).float()

    pred_pair = (
        combo_table[
            static_batch
        ]
    )

    oracle_pair = (
        combo_table[
            oracle_idx
        ]
    )

    overlap = head_overlap(
        pred_pair,
        oracle_pair,
    )

    selected_loss = (
        val_losses.gather(
            dim=-1,
            index=static_batch[
                ...,
                None,
            ],
        )
        .squeeze(-1)
    )

    oracle_loss = (
        val_losses.min(
            dim=-1
        ).values
    )

    regret = (
        selected_loss
        -
        oracle_loss
    )

    return {
        "pair":
            combo_table[
                static_idx
            ],

        "exact":
            exact.mean().item(),

        "overlap":
            overlap.mean().item(),

        "mean_regret":
            regret.mean().item(),
    }


# ============================================================
# Printing
# ============================================================

def print_metrics(
    title,
    m,
):
    print(
        f"\n{title}"
    )

    print(
        "Individual Utility only:"
    )

    print(
        f"  Oracle exact: "
        f"{100.0 * m['utility_exact']:.2f}%"
    )

    print(
        f"  Oracle overlap: "
        f"{100.0 * m['utility_overlap']:.2f}%"
    )

    print(
        f"  Mean regret: "
        f"{m['utility_mean_regret']:.8e}"
    )

    print(
        f"  Teacher Top-1 agreement: "
        f"{100.0 * m['utility_teacher_top1']:.2f}%"
    )

    print(
        f"  Teacher Top-2 exact: "
        f"{100.0 * m['utility_teacher_pair']:.2f}%"
    )

    print(
        "\nUtility + Head Interaction:"
    )

    print(
        f"  Oracle exact: "
        f"{100.0 * m['final_exact']:.2f}%"
    )

    print(
        f"  Oracle overlap: "
        f"{100.0 * m['final_overlap']:.2f}%"
    )

    print(
        f"  Mean regret: "
        f"{m['final_mean_regret']:.8e}"
    )

    print(
        f"  Median regret: "
        f"{m['final_median_regret']:.8e}"
    )

    print(
        f"  Mean |interaction|: "
        f"{m['mean_abs_interaction']:.6f}"
    )


def print_frequency(
    pred_idx,
    combo_table,
):
    print(
        "\nFinal Direct pair frequency "
        "(held-out validation)"
    )

    for block_idx in range(
        pred_idx.shape[1]
    ):
        counter = Counter(
            pred_idx[
                :,
                block_idx
            ].tolist()
        )

        total = (
            pred_idx.shape[0]
        )

        print(
            f"\nBlock {block_idx}:"
        )

        for idx, combo in enumerate(
            combo_table.tolist()
        ):
            count = counter.get(
                idx,
                0,
            )

            print(
                f"  {tuple(combo)}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
            )


# ============================================================
# Training
# ============================================================

def train(
    predictors,
    train_loader,
    val_loader,
    combo_table,
    device,
    args,
):
    parameters = [
        p
        for predictor in predictors
        for p in predictor.parameters()
    ]

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                args.epochs,
            ),
        )
    )

    before = evaluate(
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print_metrics(
        "Validation BEFORE training",
        before,
    )

    best_regret = math.inf
    best_exact = -1.0
    best_epoch = -1

    os.makedirs(
        os.path.dirname(
            args.checkpoint
        )
        or ".",
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        for predictor in predictors:
            predictor.train()

        sums = {
            "total": 0.0,
            "expected_regret": 0.0,
            "pair_kl": 0.0,
            "utility_kl": 0.0,
            "interaction_l2": 0.0,
            "n": 0,
        }

        for (
            features,
            subset_losses,
            teacher_target,
        ) in train_loader:
            features = features.to(
                device,
                non_blocking=True,
            )

            subset_losses = subset_losses.to(
                device,
                non_blocking=True,
            )

            teacher_target = teacher_target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                pair_scores,
                info,
            ) = forward_all_blocks(
                predictors,
                features,
                return_info=True,
            )

            loss, li = compute_loss(
                pair_scores=pair_scores,
                utility_logits=info[
                    "utility_logits"
                ],
                interaction_scores=info[
                    "interaction_scores"
                ],
                subset_losses=subset_losses,
                teacher_target=teacher_target,
                args=args,
            )

            loss.backward()
            optimizer.step()

            B = (
                features.shape[0]
            )

            sums["total"] += (
                loss.item()
                *
                B
            )

            for key in [
                "expected_regret",
                "pair_kl",
                "utility_kl",
                "interaction_l2",
            ]:
                sums[key] += (
                    li[key].item()
                    *
                    B
                )

            sums["n"] += B

        scheduler.step()

        val = evaluate(
            predictors,
            val_loader,
            combo_table,
            device,
        )

        n = sums["n"]

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"loss={sums['total']/n:.6f} | "
            f"reg={sums['expected_regret']/n:.6f} | "
            f"pairKL={sums['pair_kl']/n:.6f} | "
            f"utilityKL={sums['utility_kl']/n:.6f} | "
            f"intL2={sums['interaction_l2']/n:.6f}"
        )

        print(
            f"Utility-only: "
            f"exact={100.0 * val['utility_exact']:.2f}% | "
            f"overlap={100.0 * val['utility_overlap']:.2f}% | "
            f"regret={val['utility_mean_regret']:.8e}"
        )

        print(
            f"+Interaction: "
            f"exact={100.0 * val['final_exact']:.2f}% | "
            f"overlap={100.0 * val['final_overlap']:.2f}% | "
            f"regret={val['final_mean_regret']:.8e}"
        )

        better = (
            val[
                "final_mean_regret"
            ]
            <
            best_regret
            -
            1e-12
        )

        if (
            not better
            and
            abs(
                val[
                    "final_mean_regret"
                ]
                -
                best_regret
            )
            <=
            1e-12
            and
            val[
                "final_exact"
            ]
            >
            best_exact
        ):
            better = True

        if better:
            best_regret = (
                val[
                    "final_mean_regret"
                ]
            )

            best_exact = (
                val[
                    "final_exact"
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "predictors": {
                        f"block_{i}":
                            predictor.state_dict()
                        for i, predictor
                        in enumerate(
                            predictors
                        )
                    },

                    "best_epoch":
                        best_epoch,

                    "best_val_regret":
                        best_regret,

                    "best_val_exact":
                        best_exact,

                    "combination_table":
                        combo_table,

                    "config":
                        vars(args),
                },
                args.checkpoint,
            )

    best = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    for i, predictor in enumerate(
        predictors
    ):
        predictor.load_state_dict(
            best[
                "predictors"
            ][
                f"block_{i}"
            ]
        )

    after = evaluate(
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        args.checkpoint
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print_metrics(
        "FINAL HELD-OUT validation",
        after,
    )

    return before, after


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

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "Device:",
        device,
    )

    if not os.path.exists(
        args.teacher_cache
    ):
        raise FileNotFoundError(
            f"Teacher cache not found:\n{args.teacher_cache}"
        )

    if not os.path.exists(
        args.feature_cache
    ):
        raise FileNotFoundError(
            f"Feature cache not found:\n{args.feature_cache}\n"
            "Run the previous subset-value experiment first."
        )

    teacher_cache = torch.load(
        args.teacher_cache,
        map_location="cpu",
        weights_only=False,
    )

    feature_cache = torch.load(
        args.feature_cache,
        map_location="cpu",
        weights_only=False,
    )

    train_teacher = (
        teacher_cache[
            "train"
        ]
    )

    val_teacher = (
        teacher_cache[
            "val"
        ]
    )

    train_features = (
        feature_cache[
            "train"
        ].float()
    )

    val_features = (
        feature_cache[
            "val"
        ].float()
    )

    train_losses = (
        train_teacher[
            "subset_losses"
        ].float()
    )

    val_losses = (
        val_teacher[
            "subset_losses"
        ].float()
    )

    train_utility_target = (
        train_teacher[
            "teacher_target"
        ].float()
    )

    val_utility_target = (
        val_teacher[
            "teacher_target"
        ].float()
    )

    combo_table = (
        train_teacher[
            "combination_table"
        ].long()
    )

    print(
        "\nCached feature shapes:"
    )

    print(
        "train:",
        tuple(
            train_features.shape
        ),
    )

    print(
        "val:",
        tuple(
            val_features.shape
        ),
    )

    print(
        "\nDirect pair table:"
    )

    print(
        combo_table
    )

    depth = (
        train_features.shape[1]
    )

    mini_heads = (
        train_features.shape[2]
    )

    feature_dim = (
        train_features.shape[3]
    )

    expected = torch.tensor(
        list(
            itertools.combinations(
                range(
                    mini_heads
                ),
                2,
            )
        ),
        dtype=torch.long,
    )

    if not torch.equal(
        combo_table,
        expected,
    ):
        raise ValueError(
            "Combination-table order mismatch."
        )

    # ---------------------------------------------------------
    # Proper static baseline
    # ---------------------------------------------------------

    static = evaluate_static(
        train_losses,
        val_losses,
        combo_table,
    )

    print(
        "\n================ STATIC BASELINE ================"
    )

    print(
        "Fixed Direct pair per block:"
    )

    print(
        static[
            "pair"
        ]
    )

    print(
        f"Exact: "
        f"{100.0 * static['exact']:.2f}%"
    )

    print(
        f"Overlap: "
        f"{100.0 * static['overlap']:.2f}%"
    )

    print(
        f"Mean regret: "
        f"{static['mean_regret']:.8e}"
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    train_ds = CachedDataset(
        train_features,
        train_losses,
        train_utility_target,
    )

    val_ds = CachedDataset(
        val_features,
        val_losses,
        val_utility_target,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ---------------------------------------------------------
    # Predictors
    # ---------------------------------------------------------

    predictors = torch.nn.ModuleList(
        [
            UtilityInteractionPredictor(
                feature_dim=feature_dim,
                mini_heads=mini_heads,
                direct_k=2,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )

            for _ in range(
                depth
            )
        ]
    ).to(
        device
    )

    for block_idx, predictor in enumerate(
        predictors
    ):
        print(
            f"Block {block_idx} predictor params: "
            f"{sum(p.numel() for p in predictor.parameters()):,}"
        )

    before, after = train(
        predictors=predictors,
        train_loader=train_loader,
        val_loader=val_loader,
        combo_table=combo_table,
        device=device,
        args=args,
    )

    print_frequency(
        after[
            "pred_idx"
        ],
        combo_table,
    )

    print(
        "\n================ FINAL COMPARISON ================"
    )

    print(
        "Random exact expectation: 16.67%"
    )

    print(
        "Random overlap expectation: 50.00%"
    )

    print(
        "\nStatic:"
    )

    print(
        f"  exact={100.0 * static['exact']:.2f}% | "
        f"overlap={100.0 * static['overlap']:.2f}% | "
        f"regret={static['mean_regret']:.8e}"
    )

    print(
        "\nIndividual Utility only:"
    )

    print(
        f"  exact={100.0 * after['utility_exact']:.2f}% | "
        f"overlap={100.0 * after['utility_overlap']:.2f}% | "
        f"regret={after['utility_mean_regret']:.8e}"
    )

    print(
        "\nIndividual Utility + Head Interaction:"
    )

    print(
        f"  exact={100.0 * after['final_exact']:.2f}% | "
        f"overlap={100.0 * after['final_overlap']:.2f}% | "
        f"regret={after['final_mean_regret']:.8e}"
    )

    print(
        "\nThe final score is:"
    )

    print(
        "pair_score(i,j) = utility(i) + utility(j) + interaction(i,j)"
    )


if __name__ == "__main__":
    main()
