import argparse
import itertools
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.utility_interaction_predictor import (
    UtilityInteractionPredictor,
)
from models.direct_pair_value_predictor import (
    DirectPairValuePredictor,
    ContextualDirectPairValuePredictor,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--route-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage3_global_value_route_cache.pt"
        ),
    )

    p.add_argument(
        "--feature-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage3_block0_rich_feature_cache.pt"
        ),
    )

    p.add_argument(
        "--output-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage3_pair_formulation_probe.pt"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    p.add_argument(
        "--inner-val-samples",
        type=int,
        default=200,
    )

    p.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--regret-weight",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--pair-kl-weight",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--utility-kl-weight",
        type=float,
        default=0.5,
    )

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
# Cache loading
# ============================================================

def load_file(
    path,
    name,
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# Target
# ============================================================

def get_block0_oracle_costs(
    route_split,
):
    """
    For each Block-0 pair p0:

        cost[p0] = min_p1 final CE(p0, p1)

    Shape:
        [N, 6]
    """

    route_losses = route_split[
        "route_losses"
    ].float()

    return route_losses.min(
        dim=2
    ).values


def costs_to_utility_target(
    pair_costs,
    combo_table,
    mini_heads,
    eps=1e-8,
):
    utilities = []

    for head_idx in range(
        mini_heads
    ):
        included = (
            combo_table
            ==
            head_idx
        ).any(
            dim=-1
        )

        excluded = ~included

        include_cost = pair_costs[
            :,
            included,
        ].mean(
            dim=-1
        )

        exclude_cost = pair_costs[
            :,
            excluded,
        ].mean(
            dim=-1
        )

        utilities.append(
            exclude_cost
            -
            include_cost
        )

    utility = torch.stack(
        utilities,
        dim=-1,
    )

    centered = (
        utility
        -
        utility.mean(
            dim=-1,
            keepdim=True,
        )
    )

    std = centered.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    )

    normalized = (
        centered
        /
        std.clamp_min(
            eps
        )
    )

    target = torch.softmax(
        normalized,
        dim=-1,
    )

    zero_signal = (
        std.squeeze(
            -1
        )
        <=
        eps
    )

    if zero_signal.any():
        target[
            zero_signal
        ] = (
            1.0
            /
            mini_heads
        )

    return target


# ============================================================
# Dataset
# ============================================================

class FormulationDataset(Dataset):
    def __init__(
        self,
        original_head_features,
        rich_head_features,
        global_features,
        pair_features,
        pair_costs,
        utility_target,
        indices,
    ):
        self.original_head_features = (
            original_head_features[
                indices
            ].float()
        )

        self.rich_head_features = (
            rich_head_features[
                indices
            ].float()
        )

        self.global_features = (
            global_features[
                indices
            ].float()
        )

        self.pair_features = (
            pair_features[
                indices
            ].float()
        )

        self.pair_costs = (
            pair_costs[
                indices
            ].float()
        )

        self.utility_target = (
            utility_target[
                indices
            ].float()
        )

    def __len__(self):
        return self.pair_costs.shape[0]

    def __getitem__(
        self,
        idx,
    ):
        return (
            self.original_head_features[
                idx
            ],
            self.rich_head_features[
                idx
            ],
            self.global_features[
                idx
            ],
            self.pair_features[
                idx
            ],
            self.pair_costs[
                idx
            ],
            self.utility_target[
                idx
            ],
        )


def move_batch(
    batch,
    device,
):
    return tuple(
        x.to(
            device,
            non_blocking=True,
        )
        for x in batch
    )


# ============================================================
# Models
# ============================================================

def build_model(
    variant,
    feature_shapes,
    mini_heads,
    device,
):
    if variant in (
        "ui_full",
        "ui_pair_only",
    ):
        model = UtilityInteractionPredictor(
            feature_dim=feature_shapes[
                "original"
            ],
            mini_heads=mini_heads,
            direct_k=2,
            hidden_dim=64,
            dropout=0.0,
        )

    elif variant == "direct34":
        model = DirectPairValuePredictor(
            feature_dim=feature_shapes[
                "original"
            ],
            mini_heads=mini_heads,
            direct_k=2,
            hidden_dim=64,
            dropout=0.0,
        )

    elif variant == "direct_rich":
        model = ContextualDirectPairValuePredictor(
            head_feature_dim=feature_shapes[
                "rich"
            ],
            global_feature_dim=feature_shapes[
                "global"
            ],
            pair_feature_dim=feature_shapes[
                "pair"
            ],
            mini_heads=mini_heads,
            direct_k=2,
            hidden_dim=64,
            dropout=0.0,
        )

    else:
        raise ValueError(
            variant
        )

    return model.to(
        device
    )


def forward_variant(
    model,
    variant,
    batch,
):
    (
        original_head,
        rich_head,
        global_features,
        pair_features,
        pair_costs,
        utility_target,
    ) = batch

    if variant in (
        "ui_full",
        "ui_pair_only",
    ):
        return model(
            original_head,
            return_info=True,
        )

    if variant == "direct34":
        return model(
            original_head,
            return_info=True,
        )

    if variant == "direct_rich":
        return model(
            rich_head,
            global_features,
            pair_features,
            return_info=True,
        )

    raise ValueError(
        variant
    )


def trainable_params(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Loss
# ============================================================

def normalized_pair_costs(
    pair_costs,
    eps=1e-8,
):
    min_cost = pair_costs.min(
        dim=-1,
        keepdim=True,
    ).values

    max_cost = pair_costs.max(
        dim=-1,
        keepdim=True,
    ).values

    spread = (
        max_cost
        -
        min_cost
    ).clamp_min(
        eps
    )

    return (
        pair_costs
        -
        min_cost
    ) / spread


def compute_loss(
    model,
    variant,
    batch,
    args,
):
    pair_scores, info = (
        forward_variant(
            model,
            variant,
            batch,
        )
    )

    pair_costs = batch[
        4
    ]

    utility_target = batch[
        5
    ]

    normalized_cost = (
        normalized_pair_costs(
            pair_costs
        )
    )

    pair_prob = torch.softmax(
        pair_scores,
        dim=-1,
    )

    expected_regret = (
        pair_prob
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

    total = (
        args.regret_weight
        *
        expected_regret
        +
        args.pair_kl_weight
        *
        pair_kl
    )

    # Current production objective:
    # preserve individual utility supervision and interaction regularization.
    if variant == "ui_full":
        utility_kl = F.kl_div(
            F.log_softmax(
                info[
                    "utility_logits"
                ],
                dim=-1,
            ),
            utility_target,
            reduction="batchmean",
        )

        interaction_l2 = (
            info[
                "interaction_scores"
            ].pow(
                2
            ).mean()
        )

        total = (
            total
            +
            args.utility_kl_weight
            *
            utility_kl
            +
            args.interaction_l2_weight
            *
            interaction_l2
        )

    # Same U+I architecture but remove the individual-utility teacher.
    # Keep a tiny interaction regularizer so the correction term
    # does not drift arbitrarily.
    elif variant == "ui_pair_only":
        interaction_l2 = (
            info[
                "interaction_scores"
            ].pow(
                2
            ).mean()
        )

        total = (
            total
            +
            args.interaction_l2_weight
            *
            interaction_l2
        )

    # Direct pair predictors intentionally have no utility auxiliary loss.
    elif variant in (
        "direct34",
        "direct_rich",
    ):
        pass

    else:
        raise ValueError(
            variant
        )

    return total


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    variant,
    loader,
    device,
):
    model.eval()

    pred_all = []
    oracle_all = []
    selected_cost_all = []
    best_cost_all = []

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        pair_costs = batch[
            4
        ]

        pair_scores, _ = (
            forward_variant(
                model,
                variant,
                batch,
            )
        )

        pred = pair_scores.argmax(
            dim=-1
        )

        oracle = pair_costs.argmin(
            dim=-1
        )

        selected_cost = pair_costs.gather(
            dim=-1,
            index=pred[
                :,
                None,
            ],
        ).squeeze(
            -1
        )

        best_cost = pair_costs.min(
            dim=-1
        ).values

        pred_all.append(
            pred.cpu()
        )

        oracle_all.append(
            oracle.cpu()
        )

        selected_cost_all.append(
            selected_cost.cpu()
        )

        best_cost_all.append(
            best_cost.cpu()
        )

    pred = torch.cat(
        pred_all
    )

    oracle = torch.cat(
        oracle_all
    )

    selected_cost = torch.cat(
        selected_cost_all
    )

    best_cost = torch.cat(
        best_cost_all
    )

    regret = (
        selected_cost
        -
        best_cost
    )

    return {
        "exact":
            (
                pred
                ==
                oracle
            ).float().mean().item(),

        "selected_ce":
            selected_cost.mean().item(),

        "oracle_ce":
            best_cost.mean().item(),

        "mean_regret":
            regret.mean().item(),

        "median_regret":
            regret.median().item(),
    }


# ============================================================
# One repeated experiment
# ============================================================

def run_repeat(
    variant,
    repeat_seed,
    train_features,
    val_features,
    train_costs,
    val_costs,
    train_utility_target,
    val_utility_target,
    combo_table,
    mini_heads,
    device,
    args,
):
    seed_everything(
        repeat_seed
    )

    n_train = train_costs.shape[0]

    g = torch.Generator().manual_seed(
        repeat_seed
    )

    perm = torch.randperm(
        n_train,
        generator=g,
    )

    inner_val_idx = perm[
        :
        args.inner_val_samples
    ]

    inner_train_idx = perm[
        args.inner_val_samples:
    ]

    external_val_idx = torch.arange(
        val_costs.shape[0]
    )

    train_ds = FormulationDataset(
        train_features[
            "original_head_features"
        ],
        train_features[
            "rich_head_features"
        ],
        train_features[
            "global_features"
        ],
        train_features[
            "pair_features"
        ],
        train_costs,
        train_utility_target,
        inner_train_idx,
    )

    inner_val_ds = FormulationDataset(
        train_features[
            "original_head_features"
        ],
        train_features[
            "rich_head_features"
        ],
        train_features[
            "global_features"
        ],
        train_features[
            "pair_features"
        ],
        train_costs,
        train_utility_target,
        inner_val_idx,
    )

    external_val_ds = FormulationDataset(
        val_features[
            "original_head_features"
        ],
        val_features[
            "rich_head_features"
        ],
        val_features[
            "global_features"
        ],
        val_features[
            "pair_features"
        ],
        val_costs,
        val_utility_target,
        external_val_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    inner_val_loader = DataLoader(
        inner_val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    external_val_loader = DataLoader(
        external_val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    feature_shapes = {
        "original":
            train_features[
                "original_head_features"
            ].shape[-1],

        "rich":
            train_features[
                "rich_head_features"
            ].shape[-1],

        "global":
            train_features[
                "global_features"
            ].shape[-1],

        "pair":
            train_features[
                "pair_features"
            ].shape[-1],
    }

    model = build_model(
        variant=variant,
        feature_shapes=feature_shapes,
        mini_heads=mini_heads,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            args.epochs,
        ),
    )

    best_state = deepcopy(
        model.state_dict()
    )

    best_inner_regret = float(
        "inf"
    )

    best_epoch = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        for batch in train_loader:
            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_loss(
                model,
                variant,
                batch,
                args,
            )

            loss.backward()
            optimizer.step()

        scheduler.step()

        inner = evaluate(
            model,
            variant,
            inner_val_loader,
            device,
        )

        if (
            inner[
                "mean_regret"
            ]
            <
            best_inner_regret
        ):
            best_inner_regret = (
                inner[
                    "mean_regret"
                ]
            )

            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    external = evaluate(
        model,
        variant,
        external_val_loader,
        device,
    )

    external[
        "best_epoch"
    ] = best_epoch

    external[
        "params"
    ] = trainable_params(
        model
    )

    external[
        "state_dict"
    ] = deepcopy(
        model.state_dict()
    )

    return external


# ============================================================
# Reporting
# ============================================================

def mean_std(
    values,
):
    t = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return (
        t.mean().item(),
        t.std(
            unbiased=False
        ).item(),
    )


def summarize(
    title,
    results,
):
    exact_mean, exact_std = mean_std(
        [
            r[
                "exact"
            ]
            for r in results
        ]
    )

    regret_mean, regret_std = mean_std(
        [
            r[
                "mean_regret"
            ]
            for r in results
        ]
    )

    ce_mean, ce_std = mean_std(
        [
            r[
                "selected_ce"
            ]
            for r in results
        ]
    )

    epoch_mean, epoch_std = mean_std(
        [
            r[
                "best_epoch"
            ]
            for r in results
        ]
    )

    print(
        f"\n{title}"
    )

    print(
        f"  params: "
        f"{results[0]['params']:,}"
    )

    print(
        f"  exact: "
        f"{100.0 * exact_mean:.2f}% "
        f"± {100.0 * exact_std:.2f}%"
    )

    print(
        f"  mean regret: "
        f"{regret_mean:.8e} "
        f"± {regret_std:.8e}"
    )

    print(
        f"  selected continuation CE: "
        f"{ce_mean:.6f} "
        f"± {ce_std:.6f}"
    )

    print(
        f"  best epoch: "
        f"{epoch_mean:.1f} "
        f"± {epoch_std:.1f}"
    )

    return {
        "exact":
            exact_mean,

        "regret":
            regret_mean,

        "ce":
            ce_mean,
    }


def pct_reduction(
    reference,
    candidate,
):
    if reference == 0:
        return float(
            "nan"
        )

    return (
        100.0
        *
        (
            reference
            -
            candidate
        )
        /
        reference
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

    print(
        "\nGoal:"
    )

    print(
        "Separate three possible Block-0 bottlenecks:"
    )

    print(
        "  1) individual-utility auxiliary supervision,"
    )

    print(
        "  2) Utility+Interaction decomposition itself,"
    )

    print(
        "  3) pre-route feature insufficiency."
    )

    print(
        "\nNo fresh held-out or official CIFAR-10 test is used."
    )

    route_cache = load_file(
        args.route_cache,
        "Stage-3 route cache",
    )

    feature_cache = load_file(
        args.feature_cache,
        "Block-0 rich feature cache",
    )

    train_route = route_cache[
        "train"
    ]

    val_route = route_cache[
        "val"
    ]

    train_features = feature_cache[
        "train"
    ]

    val_features = feature_cache[
        "val"
    ]

    train_costs = get_block0_oracle_costs(
        train_route
    )

    val_costs = get_block0_oracle_costs(
        val_route
    )

    mini_heads = train_features[
        "original_head_features"
    ].shape[1]

    combinations = list(
        itertools.combinations(
            range(
                mini_heads
            ),
            2,
        )
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    train_utility_target = (
        costs_to_utility_target(
            train_costs,
            combo_table,
            mini_heads,
        )
    )

    val_utility_target = (
        costs_to_utility_target(
            val_costs,
            combo_table,
            mini_heads,
        )
    )

    print(
        "\nShapes:"
    )

    print(
        "  train original features:",
        tuple(
            train_features[
                "original_head_features"
            ].shape
        ),
    )

    print(
        "  train rich features:",
        tuple(
            train_features[
                "rich_head_features"
            ].shape
        ),
    )

    print(
        "  train global features:",
        tuple(
            train_features[
                "global_features"
            ].shape
        ),
    )

    print(
        "  train pair features:",
        tuple(
            train_features[
                "pair_features"
            ].shape
        ),
    )

    print(
        "  train oracle-continuation costs:",
        tuple(
            train_costs.shape
        ),
    )

    # Fair static Block-0 baseline selected from Stage-3 train.
    static_p0 = int(
        train_costs.mean(
            dim=0
        ).argmin().item()
    )

    static_selected = val_costs[
        :,
        static_p0
    ]

    val_oracle = val_costs.min(
        dim=-1
    ).values

    static_regret = (
        static_selected
        -
        val_oracle
    ).mean().item()

    print(
        "\n================ STATIC BLOCK-0 BASELINE ================"
    )

    print(
        f"Best fixed p0 from train: "
        f"{combinations[static_p0]}"
    )

    print(
        f"Val mean regret: "
        f"{static_regret:.8e}"
    )

    variants = [
        "ui_full",
        "ui_pair_only",
        "direct34",
        "direct_rich",
    ]

    titles = {
        "ui_full":
            "A. Utility+Interaction / current full objective",

        "ui_pair_only":
            "B. Utility+Interaction / pair-value objective only",

        "direct34":
            "C. Direct 6-pair predictor / same 34-D features",

        "direct_rich":
            "D. Direct 6-pair predictor / rich contextual features",
    }

    all_results = {
        variant: []
        for variant in variants
    }

    best_states = {
        variant: None
        for variant in variants
    }

    best_regrets = {
        variant: float(
            "inf"
        )
        for variant in variants
    }

    print(
        "\n================ FORMULATION PROBE TRAINING ================"
    )

    for repeat_idx in range(
        args.repeats
    ):
        repeat_seed = (
            args.seed
            +
            1000
            *
            (
                repeat_idx
                +
                1
            )
        )

        print(
            f"\n----- Repeat "
            f"{repeat_idx + 1}/"
            f"{args.repeats} "
            f"(seed={repeat_seed}) -----"
        )

        for variant in variants:
            result = run_repeat(
                variant=variant,
                repeat_seed=repeat_seed,
                train_features=train_features,
                val_features=val_features,
                train_costs=train_costs,
                val_costs=val_costs,
                train_utility_target=train_utility_target,
                val_utility_target=val_utility_target,
                combo_table=combo_table,
                mini_heads=mini_heads,
                device=device,
                args=args,
            )

            all_results[
                variant
            ].append(
                result
            )

            print(
                f"{variant:12s} | "
                f"exact={100.0 * result['exact']:.2f}% | "
                f"regret={result['mean_regret']:.8e} | "
                f"CE={result['selected_ce']:.6f} | "
                f"best_epoch={result['best_epoch']} | "
                f"params={result['params']:,}"
            )

            if (
                result[
                    "mean_regret"
                ]
                <
                best_regrets[
                    variant
                ]
            ):
                best_regrets[
                    variant
                ] = result[
                    "mean_regret"
                ]

                best_states[
                    variant
                ] = result[
                    "state_dict"
                ]

    print(
        "\n================ FORMULATION PROBE SUMMARY ================"
    )

    summaries = {}

    for variant in variants:
        summaries[
            variant
        ] = summarize(
            titles[
                variant
            ],
            all_results[
                variant
            ],
        )

    ui_full = summaries[
        "ui_full"
    ][
        "regret"
    ]

    ui_pair = summaries[
        "ui_pair_only"
    ][
        "regret"
    ]

    direct34 = summaries[
        "direct34"
    ][
        "regret"
    ]

    direct_rich = summaries[
        "direct_rich"
    ][
        "regret"
    ]

    print(
        "\n================ BOTTLENECK DECOMPOSITION ================"
    )

    print(
        f"Static regret: "
        f"{static_regret:.8e}"
    )

    print(
        f"U+I full regret: "
        f"{ui_full:.8e}"
    )

    print(
        f"U+I pair-only regret: "
        f"{ui_pair:.8e}"
    )

    print(
        f"Direct34 regret: "
        f"{direct34:.8e}"
    )

    print(
        f"DirectRich regret: "
        f"{direct_rich:.8e}"
    )

    print(
        f"\nRemoving utility auxiliary loss:"
        f" {pct_reduction(ui_full, ui_pair):+.2f}% regret reduction"
    )

    print(
        f"Removing U+I decomposition "
        f"(Direct34 vs U+I pair-only):"
        f" {pct_reduction(ui_pair, direct34):+.2f}% regret reduction"
    )

    print(
        f"Adding rich context to direct scorer:"
        f" {pct_reduction(direct34, direct_rich):+.2f}% regret reduction"
    )

    os.makedirs(
        os.path.dirname(
            args.output_checkpoint
        )
        or
        ".",
        exist_ok=True,
    )

    torch.save(
        {
            "best_states":
                best_states,

            "summaries":
                summaries,

            "static_regret":
                static_regret,

            "config": {
                "mini_heads":
                    mini_heads,

                "direct_k":
                    2,

                "hidden_dim":
                    64,

                "target":
                    "block0_oracle_continuation_value",

                "repeats":
                    args.repeats,

                "epochs":
                    args.epochs,
            },

            "source_route_cache":
                args.route_cache,

            "source_feature_cache":
                args.feature_cache,
        },
        args.output_checkpoint,
    )

    print(
        "\nSaved formulation probe:"
    )

    print(
        args.output_checkpoint
    )

    print(
        "\n================ FORMULATION-PROBE VERDICT ================"
    )

    aux_help = (
        ui_pair
        <
        0.90
        *
        ui_full
    )

    direct_help = (
        direct34
        <
        0.90
        *
        ui_pair
    )

    rich_help = (
        direct_rich
        <
        0.90
        *
        direct34
    )

    if direct_help:
        print(
            "FORMULATION BOTTLENECK SUPPORTED: "
            "with the same 34-D information, directly predicting the six pair values "
            "substantially outperforms Utility+Interaction decomposition."
        )

        print(
            "The next production-router design should keep individual utility as an auxiliary/explanatory signal, "
            "but should not force final pair selection to be only utility(i)+utility(j)+interaction(i,j)."
        )

    elif aux_help:
        print(
            "UTILITY-AUXILIARY BOTTLENECK: "
            "the U+I architecture is usable, but forcing individual utility supervision "
            "hurts the final pair-value objective."
        )

        print(
            "Keep the decomposition, but decouple explanatory utility supervision from the final routing score."
        )

    elif rich_help:
        print(
            "CONTEXT HELPS ONLY AFTER RELAXING THE SCORER: "
            "direct pair scoring benefits substantially from richer state information."
        )

        print(
            "The next router should combine a direct pair-value head with richer Block-0 context."
        )

    else:
        print(
            "PRE-ROUTE PREDICTION LIMITATION LIKELY: "
            "neither removing utility supervision, relaxing the U+I decomposition, "
            "nor adding the tested richer context yields a large regret reduction."
        )

        print(
            "Do not keep expanding this supervised pre-route predictor blindly. "
            "The next hypothesis should be joint router-backbone training or a cheap post-candidate preview signal."
        )


if __name__ == "__main__":
    main()
