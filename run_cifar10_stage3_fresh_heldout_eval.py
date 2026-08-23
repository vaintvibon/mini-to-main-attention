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

    p.add_argument("--data-dir", type=str, default="/content/cifar10")

    p.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage1_cifar10_seedscale_tuned.pt"
        ),
    )

    p.add_argument(
        "--old-predictor-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument(
        "--new-predictor-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage3_global_value_predictor.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # Previously consumed regions.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)
    p.add_argument("--diagnostic-subset", type=int, default=1000)
    p.add_argument("--scale-train-subset", type=int, default=1000)
    p.add_argument("--scale-val-subset", type=int, default=500)
    p.add_argument("--previous-heldout-subset", type=int, default=1000)
    p.add_argument("--decision-subset", type=int, default=1000)
    p.add_argument("--global-train-subset", type=int, default=1000)
    p.add_argument("--global-val-subset", type=int, default=500)

    # Static route is selected on the already-used Stage-3 train region.
    p.add_argument("--static-search-samples", type=int, default=1000)

    # Completely fresh set after Stage-3 val.
    p.add_argument("--heldout-samples", type=int, default=1000)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)

    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Loading
# ============================================================

def load_file(path, device, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found:\n{path}")

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def build_model_from_config(config):
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


def load_model(path, device):
    ckpt = load_file(
        path,
        device,
        "Backbone checkpoint",
    )

    model = build_model_from_config(
        ckpt["config"]
    ).to(device)

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def load_predictors(
    path,
    model,
    device,
):
    ckpt = load_file(
        path,
        device,
        "Predictor checkpoint",
    )

    config = ckpt.get("config", {})

    hidden_dim = int(
        config.get("hidden_dim", 64)
    )

    dropout = float(
        config.get("dropout", 0.0)
    )

    mini_head_dim = (
        model.blocks[0]
        .attn
        .mini_head_dim
    )

    feature_dim = (
        2 * mini_head_dim
        +
        2
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

    states = ckpt["predictors"]

    for block_idx, predictor in enumerate(
        predictors
    ):
        predictor.load_state_dict(
            states[f"block_{block_idx}"],
            strict=True,
        )

        predictor.eval()

        for p in predictor.parameters():
            p.requires_grad_(False)

    return predictors


# ============================================================
# Data
# ============================================================

def get_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_datasets(args):
    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=get_transform(),
    )

    g = torch.Generator().manual_seed(
        args.seed
    )

    permutation = torch.randperm(
        len(base),
        generator=g,
    ).tolist()

    # Stage-3 train begins after everything before Stage 3.
    stage3_train_start = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
        +
        args.utility_train_subset
        +
        args.utility_val_subset
        +
        args.diagnostic_subset
        +
        args.scale_train_subset
        +
        args.scale_val_subset
        +
        args.previous_heldout_subset
        +
        args.decision_subset
    )

    static_start = stage3_train_start
    static_end = (
        static_start
        +
        args.static_search_samples
    )

    # Fresh held-out begins after Stage-3 train + Stage-3 val.
    heldout_start = (
        stage3_train_start
        +
        args.global_train_subset
        +
        args.global_val_subset
    )

    heldout_end = (
        heldout_start
        +
        args.heldout_samples
    )

    if heldout_end > len(base):
        raise ValueError(
            f"Held-out split ends at {heldout_end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    static_indices = permutation[
        static_start:
        static_end
    ]

    heldout_indices = permutation[
        heldout_start:
        heldout_end
    ]

    if set(static_indices) & set(heldout_indices):
        raise RuntimeError(
            "Static-search and fresh held-out splits overlap."
        )

    print("\nData split:")
    print(
        f"  Static-search offset: "
        f"[{static_start}, {static_end})"
    )
    print(
        f"  Fresh held-out offset: "
        f"[{heldout_start}, {heldout_end})"
    )
    print(
        "  official CIFAR-10 test is NOT used."
    )

    return (
        Subset(
            base,
            static_indices,
        ),
        Subset(
            base,
            heldout_indices,
        ),
    )


# ============================================================
# Routing helpers
# ============================================================

def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    return nullcontext()


def extract_mini_features(
    mini_contexts,
    mini_attn,
    eps=1e-6,
):
    B, H, N, Dh = mini_contexts.shape

    cls_context = mini_contexts[
        :,
        :,
        0,
        :
    ]

    if N > 1:
        patch_mean = mini_contexts[
            :,
            :,
            1:,
            :
        ].mean(
            dim=2
        )
    else:
        patch_mean = cls_context

    M = mini_attn.shape[-1]

    p = mini_attn.clamp_min(eps)

    entropy = -(
        p * p.log()
    ).sum(
        dim=-1
    ).mean(
        dim=-1
    )

    normalized_entropy = (
        entropy
        /
        max(
            math.log(float(M)),
            eps,
        )
    )

    max_confidence = (
        mini_attn.max(
            dim=-1
        ).values.mean(
            dim=-1
        )
    )

    return torch.cat(
        [
            cls_context,
            patch_mean,
            normalized_entropy[..., None],
            max_confidence[..., None],
        ],
        dim=-1,
    )


def prepare_tokens(
    model,
    images,
):
    B = images.shape[0]

    x = model.patch_embed(
        images
    )

    cls_token = model.cls_token.expand(
        B,
        -1,
        -1,
    )

    x = torch.cat(
        [
            cls_token,
            x,
        ],
        dim=1,
    )

    x = x + model.pos_embed
    x = model.pos_drop(x)

    return x


def build_route_space(model):
    combinations = list(
        itertools.combinations(
            range(model.mini_heads),
            model.direct_k,
        )
    )

    route_configs = list(
        itertools.product(
            range(len(combinations)),
            repeat=model.depth,
        )
    )

    return combinations, route_configs


def forced_pairs_from_route(
    route_config,
    combo_table,
    batch_size,
    device,
):
    forced = []

    combo_table = combo_table.to(
        device
    )

    for combo_idx in route_config:
        forced.append(
            combo_table[
                int(combo_idx)
            ][None, :]
            .expand(
                batch_size,
                -1,
            )
            .clone()
        )

    return forced


@torch.no_grad()
def forward_forced(
    model,
    images,
    forced_pairs,
):
    return model(
        images,
        return_info=False,
        collect_taylor=False,
        forced_direct_indices_per_block=forced_pairs,
        forced_uniform_mix=True,
    )


@torch.no_grad()
def forward_dynamic(
    model,
    predictors,
    images,
    combo_table,
):
    x = prepare_tokens(
        model,
        images,
    )

    combo_table = combo_table.to(
        x.device
    )

    selected = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        x_norm = block.norm1(x)

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

        pair_scores, _ = predictors[
            block_idx
        ](
            features,
            return_info=True,
        )

        combo_idx = pair_scores.argmax(
            dim=-1
        )

        forced_pair = combo_table[
            combo_idx
        ]

        selected.append(
            combo_idx
        )

        x = block(
            x,
            patch_hw=model.patch_hw,
            return_info=False,
            collect_taylor=False,
            forced_direct_indices=forced_pair,
            forced_uniform_mix=True,
        )

    x = model.norm(x)

    logits = model.head(
        x[:, 0]
    )

    return (
        logits,
        torch.stack(
            selected,
            dim=1,
        ),
    )


# ============================================================
# Static route selection
# ============================================================

@torch.no_grad()
def search_best_static_route(
    model,
    loader,
    combinations,
    route_configs,
    combo_table,
    device,
    use_amp,
):
    route_loss_sum = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    seen = 0

    print(
        "\n================ STATIC ROUTE SEARCH ================"
    )

    print(
        f"Candidate whole-network routes: "
        f"{len(route_configs)}"
    )

    for images, labels in loader:
        images = images.to(
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
            forced = forced_pairs_from_route(
                route_config,
                combo_table,
                B,
                device,
            )

            with amp_context(
                device,
                use_amp,
            ):
                logits = forward_forced(
                    model,
                    images,
                    forced,
                )

            losses = F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            )

            route_loss_sum[
                route_idx
            ] += losses.sum().item()

        seen += B

        print(
            f"Static search: "
            f"{seen}/{len(loader.dataset)}"
        )

    mean_losses = (
        route_loss_sum
        /
        len(loader.dataset)
    )

    best_idx = int(
        mean_losses.argmin().item()
    )

    best_route = route_configs[
        best_idx
    ]

    print(
        "\nBest fixed route selected only from "
        "already-used Stage-3 train data:"
    )

    for block_idx, combo_idx in enumerate(
        best_route
    ):
        print(
            f"  Block {block_idx}: "
            f"{combinations[combo_idx]}"
        )

    print(
        f"  Search CE: "
        f"{mean_losses[best_idx].item():.6f}"
    )

    return best_route


# ============================================================
# Held-out evaluation
# ============================================================

@torch.no_grad()
def evaluate_static(
    model,
    loader,
    route_config,
    combo_table,
    device,
    use_amp,
):
    losses_all = []
    correct_all = []

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        forced = forced_pairs_from_route(
            route_config,
            combo_table,
            labels.shape[0],
            device,
        )

        with amp_context(
            device,
            use_amp,
        ):
            logits = forward_forced(
                model,
                images,
                forced,
            )

        losses_all.append(
            F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            ).cpu()
        )

        correct_all.append(
            (
                logits.argmax(dim=-1)
                ==
                labels
            ).cpu()
        )

    return {
        "losses":
            torch.cat(
                losses_all
            ),

        "correct":
            torch.cat(
                correct_all
            ),
    }


@torch.no_grad()
def evaluate_dynamic(
    model,
    predictors,
    loader,
    combo_table,
    device,
    use_amp,
    name,
):
    losses_all = []
    correct_all = []
    selected_all = []

    seen = 0

    for images, labels in loader:
        images = images.to(
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
            logits, selected = (
                forward_dynamic(
                    model,
                    predictors,
                    images,
                    combo_table,
                )
            )

        losses_all.append(
            F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            ).cpu()
        )

        correct_all.append(
            (
                logits.argmax(dim=-1)
                ==
                labels
            ).cpu()
        )

        selected_all.append(
            selected.cpu()
        )

        seen += labels.shape[0]

        print(
            f"{name}: "
            f"{seen}/{len(loader.dataset)}"
        )

    return {
        "losses":
            torch.cat(
                losses_all
            ),

        "correct":
            torch.cat(
                correct_all
            ),

        "selected":
            torch.cat(
                selected_all
            ),
    }


@torch.no_grad()
def evaluate_oracle(
    model,
    loader,
    route_configs,
    combo_table,
    device,
    use_amp,
):
    best_losses_all = []
    best_correct_all = []
    best_route_all = []

    seen = 0

    print(
        "\n================ GLOBAL ORACLE ================"
    )

    print(
        f"Routes per sample: "
        f"{len(route_configs)}"
    )

    for images, labels in loader:
        images = images.to(
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
            forced = forced_pairs_from_route(
                route_config,
                combo_table,
                B,
                device,
            )

            with amp_context(
                device,
                use_amp,
            ):
                logits = forward_forced(
                    model,
                    images,
                    forced,
                )

            losses = F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            )

            correct = (
                logits.argmax(dim=-1)
                ==
                labels
            )

            better = (
                losses < best_loss
            )

            best_loss = torch.where(
                better,
                losses,
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

        best_losses_all.append(
            best_loss.cpu()
        )

        best_correct_all.append(
            best_correct.cpu()
        )

        best_route_all.append(
            best_route_idx.cpu()
        )

        seen += B

        print(
            f"Oracle: "
            f"{seen}/{len(loader.dataset)}"
        )

    return {
        "losses":
            torch.cat(
                best_losses_all
            ),

        "correct":
            torch.cat(
                best_correct_all
            ),

        "route":
            torch.cat(
                best_route_all
            ),
    }


# ============================================================
# Statistics
# ============================================================

def mean_ce(result):
    return result[
        "losses"
    ].mean().item()


def accuracy(result):
    return (
        100.0
        *
        result[
            "correct"
        ].float().mean().item()
    )


def bootstrap_mean_ci(
    values,
    repeats,
    seed,
):
    values = (
        values.float().cpu()
    )

    n = values.numel()

    g = torch.Generator().manual_seed(
        seed
    )

    means = torch.empty(
        repeats,
        dtype=torch.float32,
    )

    done = 0
    chunk = 250

    while done < repeats:
        r = min(
            chunk,
            repeats - done,
        )

        idx = torch.randint(
            0,
            n,
            (r, n),
            generator=g,
        )

        means[
            done:
            done + r
        ] = values[
            idx
        ].mean(dim=1)

        done += r

    q = torch.quantile(
        means,
        torch.tensor(
            [0.025, 0.975]
        ),
    )

    return (
        q[0].item(),
        q[1].item(),
    )


def paired_compare(
    reference,
    candidate,
    repeats,
    seed,
):
    loss_delta = (
        candidate[
            "losses"
        ]
        -
        reference[
            "losses"
        ]
    )

    acc_delta = (
        candidate[
            "correct"
        ].float()
        -
        reference[
            "correct"
        ].float()
    ) * 100.0

    return {
        "delta_ce":
            loss_delta.mean().item(),

        "ce_ci":
            bootstrap_mean_ci(
                loss_delta,
                repeats,
                seed,
            ),

        "delta_acc":
            acc_delta.mean().item(),

        "acc_ci":
            bootstrap_mean_ci(
                acc_delta,
                repeats,
                seed + 1,
            ),

        "wrong_to_correct":
            (
                (~reference["correct"])
                &
                candidate["correct"]
            ).sum().item(),

        "correct_to_wrong":
            (
                reference["correct"]
                &
                (~candidate["correct"])
            ).sum().item(),
    }


def print_result(
    name,
    result,
):
    print(
        f"\n{name}"
    )

    print(
        f"  CE: "
        f"{mean_ce(result):.6f}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy(result):.2f}%"
    )


def print_comparison(
    title,
    stats,
):
    print(
        f"\n{title}"
    )

    lo, hi = stats[
        "ce_ci"
    ]

    print(
        f"  ΔCE(candidate-reference): "
        f"{stats['delta_ce']:+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.8f}, {hi:+.8f}]"
    )

    lo, hi = stats[
        "acc_ci"
    ]

    print(
        f"  ΔAccuracy: "
        f"{stats['delta_acc']:+.3f}%p"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.3f}, {hi:+.3f}]%p"
    )

    print(
        f"  Wrong -> Correct: "
        f"{stats['wrong_to_correct']}"
    )

    print(
        f"  Correct -> Wrong: "
        f"{stats['correct_to_wrong']}"
    )


def print_pair_frequency(
    title,
    selected,
    combinations,
):
    print(
        f"\n{title}"
    )

    for block_idx in range(
        selected.shape[1]
    ):
        print(
            f"  Block {block_idx}:"
        )

        counter = Counter(
            int(v)
            for v in selected[
                :,
                block_idx
            ].tolist()
        )

        total = (
            selected.shape[0]
        )

        for combo_idx, combo in enumerate(
            combinations
        ):
            count = counter.get(
                combo_idx,
                0,
            )

            print(
                f"    {combo}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
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

    print(
        "\nStage-3 fresh held-out evaluation."
    )

    print(
        "official CIFAR-10 test is NOT used."
    )

    model = load_model(
        args.backbone_checkpoint,
        device,
    )

    old_predictors = load_predictors(
        args.old_predictor_checkpoint,
        model,
        device,
    )

    new_predictors = load_predictors(
        args.new_predictor_checkpoint,
        model,
        device,
    )

    combinations, route_configs = (
        build_route_space(
            model
        )
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    static_set, heldout_set = (
        build_datasets(
            args
        )
    )

    static_loader = DataLoader(
        static_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    heldout_loader = DataLoader(
        heldout_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    best_static_route = (
        search_best_static_route(
            model=model,
            loader=static_loader,
            combinations=combinations,
            route_configs=route_configs,
            combo_table=combo_table,
            device=device,
            use_amp=use_amp,
        )
    )

    print(
        "\n================ FRESH HELD-OUT ================"
    )

    print(
        "Evaluating Static..."
    )

    static_result = evaluate_static(
        model=model,
        loader=heldout_loader,
        route_config=best_static_route,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    print(
        "Evaluating OLD Dynamic predictor..."
    )

    old_result = evaluate_dynamic(
        model=model,
        predictors=old_predictors,
        loader=heldout_loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
        name="OLD_DYNAMIC",
    )

    print(
        "Evaluating NEW Stage-3 global-value predictor..."
    )

    new_result = evaluate_dynamic(
        model=model,
        predictors=new_predictors,
        loader=heldout_loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
        name="NEW_GLOBAL_VALUE",
    )

    oracle_result = evaluate_oracle(
        model=model,
        loader=heldout_loader,
        route_configs=route_configs,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    print(
        "\n================ FRESH HELD-OUT SUMMARY ================"
    )

    print_result(
        "STATIC",
        static_result,
    )

    print_result(
        "OLD DYNAMIC",
        old_result,
    )

    print_result(
        "NEW GLOBAL-VALUE DYNAMIC",
        new_result,
    )

    print_result(
        "ORACLE",
        oracle_result,
    )

    static_vs_old = paired_compare(
        static_result,
        old_result,
        args.bootstrap_repeats,
        args.seed + 100,
    )

    static_vs_new = paired_compare(
        static_result,
        new_result,
        args.bootstrap_repeats,
        args.seed + 200,
    )

    old_vs_new = paired_compare(
        old_result,
        new_result,
        args.bootstrap_repeats,
        args.seed + 300,
    )

    static_vs_oracle = paired_compare(
        static_result,
        oracle_result,
        args.bootstrap_repeats,
        args.seed + 400,
    )

    new_vs_oracle = paired_compare(
        new_result,
        oracle_result,
        args.bootstrap_repeats,
        args.seed + 500,
    )

    print_comparison(
        "STATIC -> OLD DYNAMIC",
        static_vs_old,
    )

    print_comparison(
        "STATIC -> NEW GLOBAL-VALUE DYNAMIC",
        static_vs_new,
    )

    print_comparison(
        "OLD DYNAMIC -> NEW GLOBAL-VALUE DYNAMIC",
        old_vs_new,
    )

    print_comparison(
        "STATIC -> ORACLE",
        static_vs_oracle,
    )

    print_comparison(
        "NEW GLOBAL-VALUE DYNAMIC -> ORACLE",
        new_vs_oracle,
    )

    # --------------------------------------------------------
    # Oracle gap recovery
    # --------------------------------------------------------

    static_ce = mean_ce(
        static_result
    )

    old_ce = mean_ce(
        old_result
    )

    new_ce = mean_ce(
        new_result
    )

    oracle_ce = mean_ce(
        oracle_result
    )

    available_gap = (
        static_ce - oracle_ce
    )

    old_gain = (
        static_ce - old_ce
    )

    new_gain = (
        static_ce - new_ce
    )

    print(
        "\n================ ORACLE GAP RECOVERY ================"
    )

    print(
        f"Static - Oracle CE gap: "
        f"{available_gap:.8f}"
    )

    print(
        f"Static - Old Dynamic CE gain: "
        f"{old_gain:.8f}"
    )

    print(
        f"Static - New Dynamic CE gain: "
        f"{new_gain:.8f}"
    )

    if available_gap > 0:
        print(
            f"Old Dynamic captured oracle gap: "
            f"{100.0 * old_gain / available_gap:.2f}%"
        )

        print(
            f"New Dynamic captured oracle gap: "
            f"{100.0 * new_gain / available_gap:.2f}%"
        )

    # --------------------------------------------------------
    # Route behavior
    # --------------------------------------------------------

    print_pair_frequency(
        "OLD Dynamic pair frequency",
        old_result[
            "selected"
        ],
        combinations,
    )

    print_pair_frequency(
        "NEW Global-value pair frequency",
        new_result[
            "selected"
        ],
        combinations,
    )

    route_disagreement = (
        old_result[
            "selected"
        ]
        !=
        new_result[
            "selected"
        ]
    ).float().mean(
        dim=0
    )

    print(
        "\nOLD vs NEW route disagreement:"
    )

    for block_idx, value in enumerate(
        route_disagreement.tolist()
    ):
        print(
            f"  Block {block_idx}: "
            f"{100.0 * value:.2f}%"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    new_ce_lo, new_ce_hi = (
        static_vs_new[
            "ce_ci"
        ]
    )

    old_new_lo, old_new_hi = (
        old_vs_new[
            "ce_ci"
        ]
    )

    oracle_lo, oracle_hi = (
        static_vs_oracle[
            "ce_ci"
        ]
    )

    new_beats_static = (
        static_vs_new[
            "delta_ce"
        ] < 0
        and
        new_ce_hi < 0
    )

    new_beats_old = (
        old_vs_new[
            "delta_ce"
        ] < 0
        and
        old_new_hi < 0
    )

    oracle_value_exists = (
        static_vs_oracle[
            "delta_ce"
        ] < 0
        and
        oracle_hi < 0
    )

    print(
        "\n================ FRESH HELD-OUT VERDICT ================"
    )

    if (
        new_beats_static
        and
        new_beats_old
        and
        oracle_value_exists
    ):
        print(
            "STRONG PASS: Stage-3 global-value predictor "
            "beats both Static and the old Dynamic predictor "
            "with a clear held-out CE improvement."
        )

    elif (
        new_beats_static
        and
        oracle_value_exists
    ):
        print(
            "PASS: Stage-3 Dynamic routing clearly beats Static "
            "on fresh held-out CE."
        )

        print(
            "The gain over the old Dynamic predictor is not yet clearly separated."
        )

    elif oracle_value_exists:
        print(
            "SELECTOR STILL BOTTLENECKED: Oracle value is clearly present, "
            "but Stage-3 Dynamic does not clearly beat Static."
        )

        print(
            "Do not tune the same teacher again; inspect Block-0 predictor inputs/capacity."
        )

    else:
        print(
            "PIVOT WARNING: Even Oracle does not clearly beat Static "
            "on this fresh held-out set."
        )


if __name__ == "__main__":
    main()
