import argparse
import itertools
import math
import os
import random
from collections import Counter
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.utility_interaction_predictor import UtilityInteractionPredictor


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="/content/cifar10",
    )

    p.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage1_cifar10_balanced.pt"
        ),
    )

    p.add_argument(
        "--predictor-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_utility_interaction_predictor.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # Same deterministic split used in earlier stages.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)

    # Number of utility-train samples used to choose ONE global static route.
    p.add_argument("--static-search-samples", type=int, default=1000)

    # Completely separate official CIFAR-10 test split.
    p.add_argument("--test-samples", type=int, default=1000)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--random-trials", type=int, default=3)

    # Optional mixed precision for faster enumeration.
    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

def load_checkpoint(path, device, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def build_model_from_stage1_config(config):
    return DynamicMiniMainViT(
        img_size=32,
        patch_size=4,
        num_classes=10,

        embed_dim=config.get("embed_dim", 192),
        depth=config.get("depth", 2),

        main_heads=config.get("main_heads", 3),

        mini_heads=config.get("mini_heads", 4),
        mini_head_dim=config.get("mini_head_dim", 16),
        pool_ratio=2,

        utility_hidden_dim=64,

        direct_k=config.get("direct_k", 2),
        mix_temperature=1.0,

        bind_dim=64,
        bind_temperature=1.0,

        mlp_ratio=4.0,

        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


def load_predictors(
    checkpoint,
    model,
    feature_dim,
    device,
):
    config = checkpoint.get(
        "config",
        {},
    )

    hidden_dim = int(
        config.get(
            "hidden_dim",
            64,
        )
    )

    dropout = float(
        config.get(
            "dropout",
            0.0,
        )
    )

    predictors = torch.nn.ModuleList(
        [
            UtilityInteractionPredictor(
                feature_dim=feature_dim,
                mini_heads=model.mini_heads,
                direct_k=model.direct_k,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(model.depth)
        ]
    ).to(device)

    states = checkpoint.get(
        "predictors",
        None,
    )

    if states is None:
        raise KeyError(
            "Predictor checkpoint does not contain 'predictors'."
        )

    for block_idx, predictor in enumerate(predictors):
        key = f"block_{block_idx}"

        if key not in states:
            raise KeyError(
                f"Missing predictor state: {key}"
            )

        predictor.load_state_dict(
            states[key],
            strict=True,
        )

        predictor.eval()

    return predictors


# ============================================================
# Dataset
# ============================================================

def get_eval_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )


def build_static_search_dataset(args):
    """
    Uses only the Stage-2 utility-train portion of CIFAR-10 train split.

    This means the fixed static route is chosen WITHOUT using official test labels.
    """

    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=get_eval_transform(),
    )

    total = len(base)

    required = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
        +
        args.utility_train_subset
    )

    if required > total:
        raise ValueError(
            f"Requested split exceeds CIFAR-10 train set: "
            f"{required} > {total}"
        )

    g = torch.Generator().manual_seed(
        args.seed
    )

    permutation = torch.randperm(
        total,
        generator=g,
    ).tolist()

    start = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
    )

    indices = permutation[
        start:
        start + args.utility_train_subset
    ]

    n = min(
        args.static_search_samples,
        len(indices),
    )

    return Subset(
        base,
        indices[:n],
    )


def build_test_dataset(args):
    """
    Official CIFAR-10 test split.
    Never used for Stage-1 / Stage-2 training or checkpoint selection.
    """

    base = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=get_eval_transform(),
    )

    n = min(
        args.test_samples,
        len(base),
    )

    # Fixed deterministic prefix is fine because official test ordering
    # was never used in training.
    return Subset(
        base,
        list(range(n)),
    )


# ============================================================
# Mini feature extraction
# Must be IDENTICAL to the 34-d feature used for predictor training.
# ============================================================

def extract_mini_features(
    mini_contexts,
    mini_attn,
    eps=1e-6,
):
    """
    mini_contexts:
        [B,H,N,Dh]

    mini_attn:
        [B,H,N,M]

    returns:
        [B,H,2*Dh+2]
    """

    B, H, N, Dh = mini_contexts.shape

    cls_context = mini_contexts[
        :,
        :,
        0,
        :,
    ]

    if N > 1:
        patch_mean = (
            mini_contexts[
                :,
                :,
                1:,
                :,
            ]
            .mean(
                dim=2
            )
        )
    else:
        patch_mean = cls_context

    M = mini_attn.shape[-1]

    p = mini_attn.clamp_min(
        eps
    )

    entropy = -(
        p
        *
        p.log()
    ).sum(
        dim=-1
    ).mean(
        dim=-1
    )

    normalized_entropy = (
        entropy
        /
        max(
            math.log(
                float(M)
            ),
            eps,
        )
    )

    max_confidence = (
        mini_attn
        .max(
            dim=-1
        )
        .values
        .mean(
            dim=-1
        )
    )

    return torch.cat(
        [
            cls_context,
            patch_mean,
            normalized_entropy[
                ...,
                None,
            ],
            max_confidence[
                ...,
                None,
            ],
        ],
        dim=-1,
    )


# ============================================================
# Helpers
# ============================================================

def amp_context(
    device,
    enabled,
):
    if (
        enabled
        and
        device.type == "cuda"
    ):
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    return nullcontext()


def get_combinations(
    model,
):
    return list(
        itertools.combinations(
            range(model.mini_heads),
            model.direct_k,
        )
    )


def build_route_configurations(
    model,
):
    """
    For depth=2 and six pairs:
        6^2 = 36 whole-network fixed route configurations.
    """

    combinations = get_combinations(
        model
    )

    combo_indices = list(
        range(
            len(combinations)
        )
    )

    route_configs = list(
        itertools.product(
            combo_indices,
            repeat=model.depth,
        )
    )

    return combinations, route_configs


def combo_indices_to_forced(
    combo_indices,
    combinations,
    batch_size,
    device,
):
    forced = []

    for combo_idx in combo_indices:
        pair = torch.tensor(
            combinations[
                int(combo_idx)
            ],
            dtype=torch.long,
            device=device,
        )

        forced.append(
            pair[
                None,
                :,
            ]
            .expand(
                batch_size,
                -1,
            )
            .clone()
        )

    return forced


def pair_tensor_to_combo_index(
    pair,
    combo_table,
):
    """
    pair:
        [B,K]

    combo_table:
        [C,K]
    """

    pair = pair.sort(
        dim=-1
    ).values

    table = combo_table.sort(
        dim=-1
    ).values

    equality = (
        pair[
            :,
            None,
            :,
        ]
        ==
        table[
            None,
            :,
            :,
        ]
    ).all(
        dim=-1
    )

    if not equality.any(
        dim=-1
    ).all():
        raise RuntimeError(
            "Selected pair is missing from combination table."
        )

    return (
        equality
        .float()
        .argmax(
            dim=-1
        )
    )


# ============================================================
# Standard full forward with externally forced Direct routing.
#
# IMPORTANT:
# forced_uniform_mix=True is intentional.
# The counterfactual teacher and predictors were trained under this
# controlled condition. This experiment isolates Direct-pair quality.
# ============================================================

@torch.no_grad()
def forward_forced(
    model,
    x,
    forced_pairs_per_block,
):
    return model(
        x,
        return_info=False,
        collect_taylor=False,
        forced_direct_indices_per_block=(
            forced_pairs_per_block
        ),
        forced_uniform_mix=True,
    )


# ============================================================
# Sequential dynamic inference
# ============================================================

@torch.no_grad()
def forward_dynamic(
    model,
    predictors,
    x,
    combo_table,
    mode,
):
    """
    mode:
        "utility"
            Select Top-K from individual utility logits.

        "interaction"
            Select pair from:
                utility(i) + utility(j) + interaction(i,j)

    Selection is sequential by block:
        current block state
          -> Mini features
          -> choose Direct pair
          -> execute that block
          -> next block sees the resulting representation
    """

    if mode not in {
        "utility",
        "interaction",
    }:
        raise ValueError(
            f"Unsupported dynamic mode: {mode}"
        )

    B = x.shape[0]

    # ---------------------------------------------------------
    # Patch embedding + CLS + position embedding
    # ---------------------------------------------------------

    x = model.patch_embed(
        x
    )

    cls_token = (
        model.cls_token.expand(
            B,
            -1,
            -1,
        )
    )

    x = torch.cat(
        [
            cls_token,
            x,
        ],
        dim=1,
    )

    x = (
        x
        +
        model.pos_embed
    )

    x = model.pos_drop(
        x
    )

    selected_combo_indices = []

    combo_table_device = (
        combo_table.to(
            x.device
        )
    )

    # ---------------------------------------------------------
    # Block-by-block dynamic selection
    # ---------------------------------------------------------

    for block_idx, block in enumerate(
        model.blocks
    ):
        # Mini sees the same Pre-LN representation used by the block.
        x_norm = block.norm1(
            x
        )

        (
            mini_contexts,
            mini_attn,
        ) = block.attn.mini_attention(
            x_norm,
            patch_hw=model.patch_hw,
        )

        features = extract_mini_features(
            mini_contexts,
            mini_attn,
        )

        (
            pair_scores,
            predictor_info,
        ) = predictors[
            block_idx
        ](
            features,
            return_info=True,
        )

        utility_logits = (
            predictor_info[
                "utility_logits"
            ]
        )

        if mode == "interaction":
            combo_idx = pair_scores.argmax(
                dim=-1
            )

        else:
            topk = torch.topk(
                utility_logits,
                k=model.direct_k,
                dim=-1,
            ).indices

            combo_idx = (
                pair_tensor_to_combo_index(
                    topk,
                    combo_table_device,
                )
            )

        forced_pair = (
            combo_table_device[
                combo_idx
            ]
        )

        selected_combo_indices.append(
            combo_idx
        )

        # Execute actual Transformer block with the selected Direct pair.
        #
        # Note: Mini attention is recomputed inside the block.
        # This evaluation measures accuracy/loss, not latency.
        x = block(
            x,
            patch_hw=model.patch_hw,
            return_info=False,
            collect_taylor=False,
            forced_direct_indices=forced_pair,
            forced_uniform_mix=True,
        )

    x = model.norm(
        x
    )

    cls = x[:, 0]

    logits = model.head(
        cls
    )

    selected_combo_indices = torch.stack(
        selected_combo_indices,
        dim=1,
    )

    return (
        logits,
        selected_combo_indices,
    )


# ============================================================
# Search best GLOBAL fixed route on utility-train samples
# ============================================================

@torch.no_grad()
def search_best_global_static(
    model,
    loader,
    combinations,
    route_configs,
    device,
    use_amp,
):
    model.eval()

    total_losses = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    total_samples = 0

    print(
        "\n================ GLOBAL STATIC SEARCH ================"
    )

    print(
        f"Candidate whole-network routes: "
        f"{len(route_configs)}"
    )

    print(
        f"Search samples: "
        f"{len(loader.dataset)}"
    )

    for batch_idx, (
        x,
        labels,
    ) in enumerate(
        loader,
        start=1,
    ):
        x = x.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        B = labels.shape[0]

        for route_idx, route_config in enumerate(
            route_configs
        ):
            forced = combo_indices_to_forced(
                combo_indices=route_config,
                combinations=combinations,
                batch_size=B,
                device=device,
            )

            with amp_context(
                device,
                use_amp,
            ):
                logits = forward_forced(
                    model,
                    x,
                    forced,
                )

                per_sample_loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    reduction="none",
                )

            total_losses[
                route_idx
            ] += (
                per_sample_loss
                .sum()
                .item()
            )

        total_samples += B

        print(
            f"Static search: "
            f"{total_samples}/"
            f"{len(loader.dataset)}"
        )

    mean_losses = (
        total_losses
        /
        total_samples
    )

    best_route_index = int(
        mean_losses.argmin().item()
    )

    best_route = (
        route_configs[
            best_route_index
        ]
    )

    print(
        "\nBest fixed route:"
    )

    for block_idx, combo_idx in enumerate(
        best_route
    ):
        print(
            f"Block {block_idx}: "
            f"{combinations[combo_idx]}"
        )

    print(
        f"Utility-train mean CE: "
        f"{mean_losses[best_route_index].item():.6f}"
    )

    return best_route


# ============================================================
# Evaluation methods
# ============================================================

@torch.no_grad()
def evaluate_fixed_route(
    model,
    loader,
    combinations,
    route_config,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, labels in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        B = labels.shape[0]

        forced = combo_indices_to_forced(
            combo_indices=route_config,
            combinations=combinations,
            batch_size=B,
            device=device,
        )

        with amp_context(
            device,
            use_amp,
        ):
            logits = forward_forced(
                model,
                x,
                forced,
            )

            per_sample_loss = F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            )

        total_loss += (
            per_sample_loss.sum().item()
        )

        total_correct += (
            logits.argmax(
                dim=-1
            )
            ==
            labels
        ).sum().item()

        total_samples += B

    return {
        "loss":
            total_loss
            /
            total_samples,

        "accuracy":
            100.0
            *
            total_correct
            /
            total_samples,
    }


@torch.no_grad()
def evaluate_dynamic_method(
    model,
    predictors,
    loader,
    combo_table,
    device,
    use_amp,
    mode,
):
    model.eval()

    for predictor in predictors:
        predictor.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    selected_all = []

    for x, labels in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device,
            use_amp,
        ):
            (
                logits,
                selected,
            ) = forward_dynamic(
                model=model,
                predictors=predictors,
                x=x,
                combo_table=combo_table,
                mode=mode,
            )

            per_sample_loss = F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            )

        total_loss += (
            per_sample_loss
            .sum()
            .item()
        )

        total_correct += (
            logits.argmax(
                dim=-1
            )
            ==
            labels
        ).sum().item()

        total_samples += labels.numel()

        selected_all.append(
            selected.cpu()
        )

    return {
        "loss":
            total_loss
            /
            total_samples,

        "accuracy":
            100.0
            *
            total_correct
            /
            total_samples,

        "selected":
            torch.cat(
                selected_all,
                dim=0,
            ),
    }


@torch.no_grad()
def evaluate_random(
    model,
    loader,
    combinations,
    route_configs,
    device,
    use_amp,
    trials,
    seed,
):
    """
    Random routing:
        each sample and each block gets a random Direct pair.
    """

    losses = []
    accuracies = []

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
        device=device,
    )

    for trial in range(
        trials
    ):
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(
            seed + 1000 + trial
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, labels in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            B = labels.shape[0]

            forced = []

            for _ in range(
                model.depth
            ):
                combo_idx = torch.randint(
                    low=0,
                    high=len(combinations),
                    size=(B,),
                    generator=generator,
                ).to(
                    device
                )

                forced.append(
                    combo_table[
                        combo_idx
                    ]
                )

            with amp_context(
                device,
                use_amp,
            ):
                logits = forward_forced(
                    model,
                    x,
                    forced,
                )

                per_sample_loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    reduction="none",
                )

            total_loss += (
                per_sample_loss
                .sum()
                .item()
            )

            total_correct += (
                logits.argmax(
                    dim=-1
                )
                ==
                labels
            ).sum().item()

            total_samples += B

        losses.append(
            total_loss
            /
            total_samples
        )

        accuracies.append(
            100.0
            *
            total_correct
            /
            total_samples
        )

        print(
            f"Random trial {trial + 1}/{trials}: "
            f"CE={losses[-1]:.6f}, "
            f"acc={accuracies[-1]:.2f}%"
        )

    loss_tensor = torch.tensor(
        losses,
        dtype=torch.float64,
    )

    acc_tensor = torch.tensor(
        accuracies,
        dtype=torch.float64,
    )

    return {
        "loss_mean":
            loss_tensor.mean().item(),

        "loss_std":
            (
                loss_tensor.std(
                    unbiased=False
                ).item()
            ),

        "accuracy_mean":
            acc_tensor.mean().item(),

        "accuracy_std":
            (
                acc_tensor.std(
                    unbiased=False
                ).item()
            ),
    }


@torch.no_grad()
def evaluate_global_oracle(
    model,
    loader,
    combinations,
    route_configs,
    device,
    use_amp,
):
    """
    Label-dependent diagnostic upper bound.

    For every test sample:
        evaluate ALL whole-network route configurations
        and choose the route with the minimum CE loss.

    For depth=2:
        6^2 = 36 routes.

    This is NOT deployable inference.
    """

    model.eval()

    total_best_loss = 0.0
    total_best_correct = 0
    total_samples = 0

    selected_route_indices = []

    print(
        "\n================ GLOBAL ORACLE ================"
    )

    print(
        f"Routes per sample: "
        f"{len(route_configs)}"
    )

    for batch_idx, (
        x,
        labels,
    ) in enumerate(
        loader,
        start=1,
    ):
        x = x.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        B = labels.shape[0]

        best_loss = torch.full(
            (B,),
            float("inf"),
            device=device,
        )

        best_correct = torch.zeros(
            B,
            dtype=torch.bool,
            device=device,
        )

        best_route_idx = torch.zeros(
            B,
            dtype=torch.long,
            device=device,
        )

        for route_idx, route_config in enumerate(
            route_configs
        ):
            forced = combo_indices_to_forced(
                combo_indices=route_config,
                combinations=combinations,
                batch_size=B,
                device=device,
            )

            with amp_context(
                device,
                use_amp,
            ):
                logits = forward_forced(
                    model,
                    x,
                    forced,
                )

                per_sample_loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    reduction="none",
                )

            correct = (
                logits.argmax(
                    dim=-1
                )
                ==
                labels
            )

            better = (
                per_sample_loss
                <
                best_loss
            )

            best_loss = torch.where(
                better,
                per_sample_loss,
                best_loss,
            )

            best_correct = torch.where(
                better,
                correct,
                best_correct,
            )

            best_route_idx = torch.where(
                better,
                torch.full_like(
                    best_route_idx,
                    route_idx,
                ),
                best_route_idx,
            )

        total_best_loss += (
            best_loss.sum().item()
        )

        total_best_correct += (
            best_correct.sum().item()
        )

        total_samples += B

        selected_route_indices.append(
            best_route_idx.cpu()
        )

        print(
            f"Oracle: "
            f"{total_samples}/"
            f"{len(loader.dataset)}"
        )

    return {
        "loss":
            total_best_loss
            /
            total_samples,

        "accuracy":
            100.0
            *
            total_best_correct
            /
            total_samples,

        "route_indices":
            torch.cat(
                selected_route_indices,
                dim=0,
            ),
    }


# ============================================================
# Frequency printing
# ============================================================

def print_dynamic_frequency(
    title,
    selected_combo_indices,
    combinations,
):
    print(
        f"\n{title}"
    )

    depth = (
        selected_combo_indices.shape[1]
    )

    for block_idx in range(
        depth
    ):
        counter = Counter(
            int(v)
            for v in selected_combo_indices[
                :,
                block_idx
            ].tolist()
        )

        total = (
            selected_combo_indices.shape[0]
        )

        print(
            f"\nBlock {block_idx}:"
        )

        for combo_idx, combo in enumerate(
            combinations
        ):
            count = counter.get(
                combo_idx,
                0,
            )

            print(
                f"  {combo}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
            )


def print_oracle_frequency(
    oracle_route_indices,
    route_configs,
    combinations,
    depth,
):
    selected_per_block = torch.zeros(
        oracle_route_indices.shape[0],
        depth,
        dtype=torch.long,
    )

    for sample_idx, route_idx in enumerate(
        oracle_route_indices.tolist()
    ):
        route = route_configs[
            route_idx
        ]

        selected_per_block[
            sample_idx
        ] = torch.tensor(
            route,
            dtype=torch.long,
        )

    print_dynamic_frequency(
        "Oracle pair frequency",
        selected_per_block,
        combinations,
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

    use_amp = (
        args.amp
        and
        device.type == "cuda"
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
        "AMP:",
        use_amp,
    )

    # --------------------------------------------------------
    # Load Stage-1 backbone
    # --------------------------------------------------------

    stage1 = load_checkpoint(
        args.stage1_checkpoint,
        device,
        "Stage-1 checkpoint",
    )

    if (
        "model" not in stage1
        or
        "config" not in stage1
    ):
        raise KeyError(
            "Stage-1 checkpoint must contain 'model' and 'config'."
        )

    model = build_model_from_stage1_config(
        stage1[
            "config"
        ]
    ).to(
        device
    )

    model.load_state_dict(
        stage1[
            "model"
        ],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    print(
        "\nLoaded frozen Stage-1 backbone:"
    )

    print(
        args.stage1_checkpoint
    )

    # --------------------------------------------------------
    # Combination table
    # --------------------------------------------------------

    (
        combinations,
        route_configs,
    ) = build_route_configurations(
        model
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    print(
        "\nDirect pairs:"
    )

    print(
        combo_table
    )

    print(
        f"Whole-network route configurations: "
        f"{len(route_configs)}"
    )

    # --------------------------------------------------------
    # Load Utility + Interaction predictor
    # --------------------------------------------------------

    predictor_checkpoint = load_checkpoint(
        args.predictor_checkpoint,
        device,
        "Utility+Interaction predictor checkpoint",
    )

    # Existing Mini feature is:
    # CLS Dh + patch mean Dh + entropy + confidence.
    mini_head_dim = (
        model.blocks[
            0
        ].attn.mini_head_dim
    )

    feature_dim = (
        2 * mini_head_dim
        +
        2
    )

    predictors = load_predictors(
        checkpoint=predictor_checkpoint,
        model=model,
        feature_dim=feature_dim,
        device=device,
    )

    print(
        "\nLoaded Utility + Interaction predictors:"
    )

    print(
        args.predictor_checkpoint
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    static_dataset = build_static_search_dataset(
        args
    )

    test_dataset = build_test_dataset(
        args
    )

    static_loader = DataLoader(
        static_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    print(
        "\nDataset sizes:"
    )

    print(
        f"Static-search train samples: "
        f"{len(static_dataset)}"
    )

    print(
        f"Official test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # 1. Best global fixed route
    # --------------------------------------------------------

    best_static_route = search_best_global_static(
        model=model,
        loader=static_loader,
        combinations=combinations,
        route_configs=route_configs,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # 2. Actual official-test comparison
    # --------------------------------------------------------

    print(
        "\n================ FINAL ROUTING TEST ================"
    )

    random_metrics = evaluate_random(
        model=model,
        loader=test_loader,
        combinations=combinations,
        route_configs=route_configs,
        device=device,
        use_amp=use_amp,
        trials=args.random_trials,
        seed=args.seed,
    )

    print(
        "\nEvaluating fixed static routing..."
    )

    static_metrics = evaluate_fixed_route(
        model=model,
        loader=test_loader,
        combinations=combinations,
        route_config=best_static_route,
        device=device,
        use_amp=use_amp,
    )

    print(
        "\nEvaluating individual-utility dynamic routing..."
    )

    utility_metrics = evaluate_dynamic_method(
        model=model,
        predictors=predictors,
        loader=test_loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
        mode="utility",
    )

    print(
        "\nEvaluating utility + interaction dynamic routing..."
    )

    interaction_metrics = evaluate_dynamic_method(
        model=model,
        predictors=predictors,
        loader=test_loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
        mode="interaction",
    )

    oracle_metrics = evaluate_global_oracle(
        model=model,
        loader=test_loader,
        combinations=combinations,
        route_configs=route_configs,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n================ FINAL SUMMARY ================"
    )

    print(
        "Controlled condition: "
        "Remaining Mini Heads use UNIFORM Mix for every method."
    )

    print(
        "Therefore this experiment isolates the quality of Direct selection."
    )

    print(
        "\nRandom routing:"
    )

    print(
        f"  CE: "
        f"{random_metrics['loss_mean']:.6f} "
        f"± {random_metrics['loss_std']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{random_metrics['accuracy_mean']:.2f}% "
        f"± {random_metrics['accuracy_std']:.2f}%"
    )

    print(
        "\nBest fixed static routing:"
    )

    for block_idx, combo_idx in enumerate(
        best_static_route
    ):
        print(
            f"  Block {block_idx}: "
            f"{combinations[combo_idx]}"
        )

    print(
        f"  CE: "
        f"{static_metrics['loss']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{static_metrics['accuracy']:.2f}%"
    )

    print(
        "\nIndividual Utility dynamic routing:"
    )

    print(
        f"  CE: "
        f"{utility_metrics['loss']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{utility_metrics['accuracy']:.2f}%"
    )

    print(
        "\nUtility + Head Interaction dynamic routing:"
    )

    print(
        f"  CE: "
        f"{interaction_metrics['loss']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{interaction_metrics['accuracy']:.2f}%"
    )

    print(
        "\nGlobal Oracle routing "
        "(uses test labels; diagnostic upper bound only):"
    )

    print(
        f"  CE: "
        f"{oracle_metrics['loss']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{oracle_metrics['accuracy']:.2f}%"
    )

    # --------------------------------------------------------
    # Pair distributions
    # --------------------------------------------------------

    print_dynamic_frequency(
        "Individual Utility pair frequency",
        utility_metrics[
            "selected"
        ],
        combinations,
    )

    print_dynamic_frequency(
        "Utility + Interaction pair frequency",
        interaction_metrics[
            "selected"
        ],
        combinations,
    )

    print_oracle_frequency(
        oracle_route_indices=oracle_metrics[
            "route_indices"
        ],
        route_configs=route_configs,
        combinations=combinations,
        depth=model.depth,
    )

    print(
        "\n================ INTERPRETATION CHECK ================"
    )

    dynamic_better_loss = (
        interaction_metrics[
            "loss"
        ]
        <
        static_metrics[
            "loss"
        ]
    )

    dynamic_better_acc = (
        interaction_metrics[
            "accuracy"
        ]
        >
        static_metrics[
            "accuracy"
        ]
    )

    print(
        "Utility+Interaction CE better than Static:",
        dynamic_better_loss,
    )

    print(
        "Utility+Interaction Accuracy better than Static:",
        dynamic_better_acc,
    )

    if (
        dynamic_better_loss
        or
        dynamic_better_acc
    ):
        print(
            "\nAt least one real classification metric improves over "
            "the best fixed route."
        )
    else:
        print(
            "\nThe dynamic selector does not yet improve real classification "
            "over the best fixed route under this controlled setup."
        )


if __name__ == "__main__":
    main()
