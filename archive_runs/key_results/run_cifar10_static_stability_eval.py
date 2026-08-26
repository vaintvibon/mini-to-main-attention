import argparse
import itertools
import math
import os
import random
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    p.add_argument("--amp", action="store_true")

    # Exact offsets from the experiment history.
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
    p.add_argument("--heldout-subset", type=int, default=1000)

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model / predictor loading
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
# Data split
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


def build_named_splits(args):
    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=get_transform(),
    )

    g = torch.Generator().manual_seed(args.seed)

    permutation = torch.randperm(
        len(base),
        generator=g,
    ).tolist()

    s1_train_start = 0
    s1_train_end = args.stage1_train_subset

    utility_train_start = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
    )
    utility_train_end = (
        utility_train_start
        +
        args.utility_train_subset
    )

    scale_train_start = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
        +
        args.utility_train_subset
        +
        args.utility_val_subset
        +
        args.diagnostic_subset
    )
    scale_train_end = (
        scale_train_start
        +
        args.scale_train_subset
    )

    stage3_train_start = (
        scale_train_end
        +
        args.scale_val_subset
        +
        args.previous_heldout_subset
        +
        args.decision_subset
    )
    stage3_train_end = (
        stage3_train_start
        +
        args.global_train_subset
    )

    stage3_val_end = (
        stage3_train_end
        +
        args.global_val_subset
    )

    heldout_start = stage3_val_end
    heldout_end = (
        heldout_start
        +
        args.heldout_subset
    )

    if heldout_end > len(base):
        raise ValueError(
            f"Requested held-out ends at {heldout_end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    selection_ranges = {
        "Stage1Train": (
            s1_train_start,
            s1_train_end,
        ),
        "UtilityTrain": (
            utility_train_start,
            utility_train_end,
        ),
        "SeedScaleTrain": (
            scale_train_start,
            scale_train_end,
        ),
        "Stage3Train": (
            stage3_train_start,
            stage3_train_end,
        ),
    }

    selection_sets = {}

    print("\nStatic selection pools (all already-used development data):")

    for name, (start, end) in selection_ranges.items():
        print(
            f"  {name:14s}: "
            f"[{start}, {end}) "
            f"n={end-start}"
        )

        selection_sets[name] = Subset(
            base,
            permutation[start:end],
        )

    print(
        f"\nReused held-out for final comparison: "
        f"[{heldout_start}, {heldout_end}) "
        f"n={heldout_end-heldout_start}"
    )
    print(
        "This is the SAME held-out used in the immediately preceding Stage-3 evaluation."
    )
    print(
        "No new data are consumed. official CIFAR-10 test is NOT used."
    )

    heldout_set = Subset(
        base,
        permutation[heldout_start:heldout_end],
    )

    return selection_sets, heldout_set


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

    cls_context = mini_contexts[:, :, 0, :]

    if N > 1:
        patch_mean = (
            mini_contexts[:, :, 1:, :]
            .mean(dim=2)
        )
    else:
        patch_mean = cls_context

    M = mini_attn.shape[-1]

    p = mini_attn.clamp_min(eps)

    entropy = -(
        p * p.log()
    ).sum(dim=-1).mean(dim=-1)

    normalized_entropy = (
        entropy
        /
        max(
            math.log(float(M)),
            eps,
        )
    )

    max_confidence = (
        mini_attn.max(dim=-1).values
        .mean(dim=-1)
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


def prepare_tokens(model, images):
    B = images.shape[0]

    x = model.patch_embed(images)

    cls_token = model.cls_token.expand(
        B,
        -1,
        -1,
    )

    x = torch.cat(
        [cls_token, x],
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
    table = combo_table.to(device)

    return [
        table[
            int(combo_idx)
        ][None, :]
        .expand(
            batch_size,
            -1,
        )
        .clone()
        for combo_idx in route_config
    ]


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

    table = combo_table.to(
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

        forced_pair = table[
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
# Static route profiling
# ============================================================

@torch.no_grad()
def profile_all_routes(
    model,
    dataset,
    route_configs,
    combo_table,
    device,
    batch_size,
    num_workers,
    use_amp,
    name,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    loss_sum = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    correct_sum = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    seen = 0

    print(
        f"\nProfiling all 36 fixed routes: {name}"
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

            correct = (
                logits.argmax(dim=-1)
                ==
                labels
            )

            loss_sum[
                route_idx
            ] += losses.sum().item()

            correct_sum[
                route_idx
            ] += correct.sum().item()

        seen += B

        print(
            f"{name}: "
            f"{seen}/{len(dataset)}"
        )

    n = len(dataset)

    return {
        "loss_sum":
            loss_sum,

        "correct_sum":
            correct_sum,

        "n":
            n,

        "mean_ce":
            loss_sum / n,

        "accuracy":
            100.0 * correct_sum / n,
    }


def rank_vector(values):
    """
    Lower CE = better rank.
    0 is best.
    """
    order = torch.argsort(
        values
    )

    ranks = torch.empty_like(
        order,
        dtype=torch.float64,
    )

    ranks[
        order
    ] = torch.arange(
        len(values),
        dtype=torch.float64,
    )

    return ranks


def spearman_from_values(
    a,
    b,
):
    ra = rank_vector(
        a
    )

    rb = rank_vector(
        b
    )

    ra = (
        ra
        -
        ra.mean()
    )

    rb = (
        rb
        -
        rb.mean()
    )

    denom = (
        ra.pow(2).sum().sqrt()
        *
        rb.pow(2).sum().sqrt()
    )

    if denom.item() == 0:
        return float("nan")

    return (
        (ra * rb).sum()
        /
        denom
    ).item()


def topk_overlap(
    a,
    b,
    k=5,
):
    top_a = set(
        torch.argsort(
            a
        )[:k].tolist()
    )

    top_b = set(
        torch.argsort(
            b
        )[:k].tolist()
    )

    return (
        len(
            top_a & top_b
        )
        /
        k
    )


def route_to_string(
    route_config,
    combinations,
):
    return " / ".join(
        f"B{block_idx}:{combinations[combo_idx]}"
        for block_idx, combo_idx in enumerate(
            route_config
        )
    )


# ============================================================
# Held-out all-route matrix
# ============================================================

@torch.no_grad()
def evaluate_all_routes_heldout(
    model,
    dataset,
    route_configs,
    combo_table,
    device,
    batch_size,
    num_workers,
    use_amp,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    losses_all = []
    correct_all = []

    seen = 0

    print(
        "\nEvaluating all 36 fixed routes on reused held-out..."
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

        batch_losses = []
        batch_correct = []

        for route_config in route_configs:
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

            batch_losses.append(
                F.cross_entropy(
                    logits.float(),
                    labels,
                    reduction="none",
                ).cpu()
            )

            batch_correct.append(
                (
                    logits.argmax(dim=-1)
                    ==
                    labels
                ).cpu()
            )

        losses_all.append(
            torch.stack(
                batch_losses,
                dim=1,
            )
        )

        correct_all.append(
            torch.stack(
                batch_correct,
                dim=1,
            )
        )

        seen += B

        print(
            f"Held-out all routes: "
            f"{seen}/{len(dataset)}"
        )

    return {
        "losses":
            torch.cat(
                losses_all,
                dim=0,
            ),

        "correct":
            torch.cat(
                correct_all,
                dim=0,
            ),
    }


@torch.no_grad()
def evaluate_dynamic(
    model,
    predictors,
    dataset,
    combo_table,
    device,
    batch_size,
    num_workers,
    use_amp,
    name,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    losses_all = []
    correct_all = []

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
            logits, _ = forward_dynamic(
                model,
                predictors,
                images,
                combo_table,
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

        seen += labels.shape[0]

        print(
            f"{name}: "
            f"{seen}/{len(dataset)}"
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


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(
    values,
    repeats,
    seed,
):
    values = values.float().cpu()

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
        candidate["losses"]
        -
        reference["losses"]
    )

    acc_delta = (
        candidate["correct"].float()
        -
        reference["correct"].float()
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
    }


def result_from_route_column(
    heldout_matrix,
    route_idx,
):
    return {
        "losses":
            heldout_matrix[
                "losses"
            ][
                :,
                route_idx
            ],

        "correct":
            heldout_matrix[
                "correct"
            ][
                :,
                route_idx
            ],
    }


def result_oracle(
    heldout_matrix,
):
    losses = heldout_matrix[
        "losses"
    ]

    correct = heldout_matrix[
        "correct"
    ]

    best_idx = losses.argmin(
        dim=1
    )

    row = torch.arange(
        losses.shape[0]
    )

    return {
        "losses":
            losses[
                row,
                best_idx,
            ],

        "correct":
            correct[
                row,
                best_idx,
            ],

        "route_idx":
            best_idx,
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
        f"{result['losses'].mean().item():.6f}"
    )

    print(
        f"  Accuracy: "
        f"{100.0 * result['correct'].float().mean().item():.2f}%"
    )


def print_comparison(
    title,
    stats,
):
    print(
        f"\n{title}"
    )

    ce_lo, ce_hi = stats[
        "ce_ci"
    ]

    acc_lo, acc_hi = stats[
        "acc_ci"
    ]

    print(
        f"  ΔCE(candidate-reference): "
        f"{stats['delta_ce']:+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{ce_lo:+.8f}, {ce_hi:+.8f}]"
    )

    print(
        f"  ΔAccuracy: "
        f"{stats['delta_acc']:+.3f}%p"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{acc_lo:+.3f}, {acc_hi:+.3f}]%p"
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
        "\nGoal: test whether the fixed-route baseline itself is unstable."
    )

    print(
        "No new held-out data will be consumed."
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

    selection_sets, heldout_set = (
        build_named_splits(
            args
        )
    )

    # --------------------------------------------------------
    # Profile the 36 static routes on several independent
    # already-used development pools.
    # --------------------------------------------------------

    profiles = {}

    for name, dataset in selection_sets.items():
        profiles[name] = profile_all_routes(
            model=model,
            dataset=dataset,
            route_configs=route_configs,
            combo_table=combo_table,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
            name=name,
        )

    print(
        "\n================ STATIC BEST ROUTE BY POOL ================"
    )

    best_indices = {}

    for name, profile in profiles.items():
        best_idx = int(
            profile[
                "mean_ce"
            ].argmin().item()
        )

        best_indices[name] = best_idx

        sorted_ce = torch.sort(
            profile[
                "mean_ce"
            ]
        ).values

        margin = (
            sorted_ce[1]
            -
            sorted_ce[0]
        ).item()

        print(
            f"{name:14s} | "
            f"{route_to_string(route_configs[best_idx], combinations)} | "
            f"CE={profile['mean_ce'][best_idx].item():.6f} | "
            f"margin_to_2nd={margin:.8f}"
        )

    # --------------------------------------------------------
    # Ranking stability across pools.
    # --------------------------------------------------------

    names = list(
        profiles.keys()
    )

    print(
        "\n================ ROUTE RANKING STABILITY ================"
    )

    for i in range(
        len(names)
    ):
        for j in range(
            i + 1,
            len(names),
        ):
            a = names[i]
            b = names[j]

            rho = spearman_from_values(
                profiles[a]["mean_ce"],
                profiles[b]["mean_ce"],
            )

            overlap5 = topk_overlap(
                profiles[a]["mean_ce"],
                profiles[b]["mean_ce"],
                k=5,
            )

            print(
                f"{a:14s} vs {b:14s} | "
                f"Spearman={rho:+.4f} | "
                f"Top-5 overlap={100.0 * overlap5:.1f}%"
            )

    # --------------------------------------------------------
    # Robust fixed routes.
    # 1) sample-weighted union of all selection pools
    # 2) split-balanced average of per-pool mean CE
    # --------------------------------------------------------

    total_loss_sum = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    total_n = 0

    split_mean_stack = []

    for profile in profiles.values():
        total_loss_sum += profile[
            "loss_sum"
        ]

        total_n += profile[
            "n"
        ]

        split_mean_stack.append(
            profile[
                "mean_ce"
            ]
        )

    weighted_mean = (
        total_loss_sum
        /
        total_n
    )

    balanced_mean = torch.stack(
        split_mean_stack,
        dim=0,
    ).mean(
        dim=0
    )

    weighted_best_idx = int(
        weighted_mean.argmin().item()
    )

    balanced_best_idx = int(
        balanced_mean.argmin().item()
    )

    print(
        "\n================ ROBUST STATIC ROUTES ================"
    )

    print(
        f"Sample-weighted union (n={total_n}):"
    )

    print(
        "  "
        +
        route_to_string(
            route_configs[
                weighted_best_idx
            ],
            combinations,
        )
    )

    print(
        f"  development CE="
        f"{weighted_mean[weighted_best_idx].item():.6f}"
    )

    print(
        "Split-balanced average:"
    )

    print(
        "  "
        +
        route_to_string(
            route_configs[
                balanced_best_idx
            ],
            combinations,
        )
    )

    print(
        f"  balanced score="
        f"{balanced_mean[balanced_best_idx].item():.6f}"
    )

    # --------------------------------------------------------
    # Reuse SAME held-out. Evaluate all fixed routes there.
    # --------------------------------------------------------

    heldout_matrix = evaluate_all_routes_heldout(
        model=model,
        dataset=heldout_set,
        route_configs=route_configs,
        combo_table=combo_table,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
    )

    heldout_mean_ce = heldout_matrix[
        "losses"
    ].mean(
        dim=0
    )

    heldout_best_idx = int(
        heldout_mean_ce.argmin().item()
    )

    heldout_sorted = torch.argsort(
        heldout_mean_ce
    )

    weighted_rank = int(
        (
            heldout_sorted
            ==
            weighted_best_idx
        ).nonzero(
            as_tuple=False
        )[0, 0].item()
    ) + 1

    balanced_rank = int(
        (
            heldout_sorted
            ==
            balanced_best_idx
        ).nonzero(
            as_tuple=False
        )[0, 0].item()
    ) + 1

    print(
        "\n================ HELD-OUT STATIC DIAGNOSTIC ================"
    )

    print(
        "Held-out-optimal fixed route below is DIAGNOSTIC ONLY; "
        "it uses held-out labels and is not a fair baseline."
    )

    print(
        "  Held-out-optimal: "
        +
        route_to_string(
            route_configs[
                heldout_best_idx
            ],
            combinations,
        )
    )

    print(
        f"  Held-out-optimal fixed CE="
        f"{heldout_mean_ce[heldout_best_idx].item():.6f}"
    )

    print(
        f"  Sample-weighted robust route held-out rank: "
        f"{weighted_rank}/36"
    )

    print(
        f"  Split-balanced robust route held-out rank: "
        f"{balanced_rank}/36"
    )

    # --------------------------------------------------------
    # Dynamic results on the SAME held-out.
    # --------------------------------------------------------

    old_dynamic = evaluate_dynamic(
        model=model,
        predictors=old_predictors,
        dataset=heldout_set,
        combo_table=combo_table,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        name="OLD_DYNAMIC",
    )

    new_dynamic = evaluate_dynamic(
        model=model,
        predictors=new_predictors,
        dataset=heldout_set,
        combo_table=combo_table,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        name="NEW_STAGE3_DYNAMIC",
    )

    weighted_static = result_from_route_column(
        heldout_matrix,
        weighted_best_idx,
    )

    balanced_static = result_from_route_column(
        heldout_matrix,
        balanced_best_idx,
    )

    heldout_optimal_static = result_from_route_column(
        heldout_matrix,
        heldout_best_idx,
    )

    oracle = result_oracle(
        heldout_matrix
    )

    print(
        "\n================ SAME HELD-OUT SUMMARY ================"
    )

    print_result(
        "ROBUST STATIC (sample-weighted)",
        weighted_static,
    )

    print_result(
        "ROBUST STATIC (split-balanced)",
        balanced_static,
    )

    print_result(
        "HELD-OUT-OPTIMAL STATIC (diagnostic only)",
        heldout_optimal_static,
    )

    print_result(
        "OLD DYNAMIC",
        old_dynamic,
    )

    print_result(
        "NEW STAGE-3 DYNAMIC",
        new_dynamic,
    )

    print_result(
        "ORACLE",
        oracle,
    )

    # --------------------------------------------------------
    # Paired tests against the robust static baseline.
    # --------------------------------------------------------

    weighted_vs_old = paired_compare(
        weighted_static,
        old_dynamic,
        args.bootstrap_repeats,
        args.seed + 100,
    )

    weighted_vs_new = paired_compare(
        weighted_static,
        new_dynamic,
        args.bootstrap_repeats,
        args.seed + 200,
    )

    balanced_vs_new = paired_compare(
        balanced_static,
        new_dynamic,
        args.bootstrap_repeats,
        args.seed + 300,
    )

    optimal_static_vs_new = paired_compare(
        heldout_optimal_static,
        new_dynamic,
        args.bootstrap_repeats,
        args.seed + 400,
    )

    weighted_vs_oracle = paired_compare(
        weighted_static,
        oracle,
        args.bootstrap_repeats,
        args.seed + 500,
    )

    print_comparison(
        "ROBUST STATIC(weighted) -> OLD DYNAMIC",
        weighted_vs_old,
    )

    print_comparison(
        "ROBUST STATIC(weighted) -> NEW STAGE-3 DYNAMIC",
        weighted_vs_new,
    )

    print_comparison(
        "ROBUST STATIC(split-balanced) -> NEW STAGE-3 DYNAMIC",
        balanced_vs_new,
    )

    print_comparison(
        "HELD-OUT-OPTIMAL STATIC(diagnostic) -> NEW DYNAMIC",
        optimal_static_vs_new,
    )

    print_comparison(
        "ROBUST STATIC(weighted) -> ORACLE",
        weighted_vs_oracle,
    )

    # --------------------------------------------------------
    # Final interpretation helper.
    # --------------------------------------------------------

    weighted_new_hi = (
        weighted_vs_new[
            "ce_ci"
        ][1]
    )

    robust_dynamic_win = (
        weighted_vs_new[
            "delta_ce"
        ] < 0
        and
        weighted_new_hi < 0
    )

    best_routes_unique = len(
        set(
            best_indices.values()
        )
    )

    pairwise_rhos = []

    for i in range(
        len(names)
    ):
        for j in range(
            i + 1,
            len(names),
        ):
            pairwise_rhos.append(
                spearman_from_values(
                    profiles[
                        names[i]
                    ][
                        "mean_ce"
                    ],
                    profiles[
                        names[j]
                    ][
                        "mean_ce"
                    ],
                )
            )

    mean_rho = sum(
        pairwise_rhos
    ) / len(
        pairwise_rhos
    )

    print(
        "\n================ STATIC-STABILITY VERDICT ================"
    )

    print(
        f"Unique best fixed routes across development pools: "
        f"{best_routes_unique}/{len(names)}"
    )

    print(
        f"Mean pairwise route-rank Spearman: "
        f"{mean_rho:+.4f}"
    )

    if robust_dynamic_win:
        print(
            "PASS: Dynamic still clearly beats a robust fixed route "
            "selected from several already-used development pools."
        )

        print(
            "The previous Dynamic>Static result is not explained away "
            "by a single unstable 1000-sample Static search."
        )
    else:
        print(
            "CAUTION: Once Static is selected robustly across several pools, "
            "Dynamic no longer clearly beats it."
        )

        print(
            "The earlier Dynamic>Static result was at least partly sensitive "
            "to fixed-route selection instability."
        )


if __name__ == "__main__":
    main()
