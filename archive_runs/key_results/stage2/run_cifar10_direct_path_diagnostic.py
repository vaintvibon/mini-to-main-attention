import argparse
import itertools
import math
import os
import random
from collections import defaultdict
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
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # Previously used train-split portions.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)

    # NEW held-out diagnostic split, taken after all previous train/val samples.
    p.add_argument("--diagnostic-samples", type=int, default=1000)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    # Seed-strength sensitivity sweep.
    p.add_argument(
        "--seed-scale-multipliers",
        type=str,
        default="0,0.5,1,2,4",
    )

    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(text):
    values = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if value < 0:
            raise ValueError(
                "seed-scale multipliers must be non-negative."
            )

        values.append(value)

    if not values:
        raise ValueError(
            "--seed-scale-multipliers cannot be empty."
        )

    if 1.0 not in values:
        values.append(1.0)

    return values


# ============================================================
# Model / predictor loading
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

    for block_idx, predictor in enumerate(
        predictors
    ):
        key = f"block_{block_idx}"

        predictor.load_state_dict(
            states[key],
            strict=True,
        )

        predictor.eval()

    return predictors


# ============================================================
# Data
# ============================================================

def build_diagnostic_dataset(args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )

    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    g = torch.Generator().manual_seed(
        args.seed
    )

    permutation = torch.randperm(
        len(base),
        generator=g,
    ).tolist()

    start = (
        args.stage1_train_subset
        +
        args.stage1_val_subset
        +
        args.utility_train_subset
        +
        args.utility_val_subset
    )

    end = (
        start
        +
        args.diagnostic_samples
    )

    if end > len(base):
        raise ValueError(
            f"Diagnostic split ends at {end}, "
            f"but CIFAR-10 train has only {len(base)} samples."
        )

    indices = permutation[
        start:end
    ]

    return Subset(
        base,
        indices,
    )


# ============================================================
# Feature extraction / routing
# ============================================================

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

    x = (
        x
        +
        model.pos_embed
    )

    x = model.pos_drop(
        x
    )

    return x


def pair_tensor_to_combo_index(
    pair,
    combo_table,
):
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


@torch.no_grad()
def dynamic_forward_with_info(
    model,
    predictors,
    images,
    combo_table,
):
    """
    Sequential dynamic routing under the refined predictor.

    Returns:
        logits
        selected_indices [B, depth]
        block_infos
    """

    x = prepare_tokens(
        model,
        images,
    )

    combo_device = combo_table.to(
        x.device
    )

    selected = []
    block_infos = []

    for block_idx, block in enumerate(
        model.blocks
    ):
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

        pair_scores, _ = predictors[
            block_idx
        ](
            features,
            return_info=True,
        )

        combo_idx = pair_scores.argmax(
            dim=-1
        )

        forced_pair = combo_device[
            combo_idx
        ]

        selected.append(
            combo_idx
        )

        # Recompute attention inside block so info exactly matches
        # the actual block execution.
        x, info = block(
            x,
            patch_hw=model.patch_hw,
            return_info=True,
            collect_taylor=False,
            forced_direct_indices=forced_pair,
            forced_uniform_mix=True,
        )

        block_infos.append(
            info
        )

    x = model.norm(
        x
    )

    logits = model.head(
        x[:, 0]
    )

    return (
        logits,
        torch.stack(
            selected,
            dim=1,
        ),
        block_infos,
    )


# ============================================================
# Mini-head diversity
# ============================================================

def pairwise_cosine_values(
    vectors,
):
    """
    vectors: [B,H,D]
    returns:
        pairwise dict {(i,j): [B]}
    """

    z = F.normalize(
        vectors.float(),
        dim=-1,
        eps=1e-8,
    )

    H = z.shape[1]

    out = {}

    for i in range(H):
        for j in range(
            i + 1,
            H,
        ):
            out[
                (i, j)
            ] = (
                z[:, i, :]
                *
                z[:, j, :]
            ).sum(
                dim=-1
            )

    return out


def collect_mini_similarity(
    info,
):
    mini_contexts = info[
        "mini_contexts"
    ].float()

    B, H, N, Dh = (
        mini_contexts.shape
    )

    cls = mini_contexts[
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
        patch_mean = cls

    full_flat = mini_contexts.reshape(
        B,
        H,
        -1,
    )

    return {
        "cls":
            pairwise_cosine_values(
                cls
            ),

        "patch":
            pairwise_cosine_values(
                patch_mean
            ),

        "full":
            pairwise_cosine_values(
                full_flat
            ),
    }


# ============================================================
# Seed contribution strength
# ============================================================

def broadcast_seed_scale(
    seed_scale,
    target,
):
    """
    target: [B,Hmain,N,Dmain]
    """

    scale = seed_scale

    if not torch.is_tensor(
        scale
    ):
        scale = torch.tensor(
            scale,
            dtype=target.dtype,
            device=target.device,
        )

    scale = scale.to(
        dtype=target.dtype,
        device=target.device,
    )

    if scale.dim() == 0:
        return scale.view(
            1,
            1,
            1,
            1,
        )

    if scale.dim() == 1:
        return scale.view(
            1,
            -1,
            1,
            1,
        )

    # Already broadcastable.
    while scale.dim() < 4:
        scale = scale.unsqueeze(
            -1
        )

    return scale


def collect_seed_strength(
    info,
):
    main_seeds = info[
        "main_seeds"
    ].float()

    seeded_q = info[
        "seeded_q"
    ].float()

    seed_scale = info[
        "seed_scale"
    ]

    scale = broadcast_seed_scale(
        seed_scale,
        main_seeds,
    )

    seed_contribution = (
        main_seeds
        *
        scale
    )

    q_base = (
        seeded_q
        -
        seed_contribution
    )

    # Frobenius norm over tokens and head dimension.
    seed_norm = torch.linalg.vector_norm(
        seed_contribution,
        dim=(-2, -1),
    )

    q_base_norm = torch.linalg.vector_norm(
        q_base,
        dim=(-2, -1),
    )

    seeded_q_norm = torch.linalg.vector_norm(
        seeded_q,
        dim=(-2, -1),
    )

    ratio = (
        seed_norm
        /
        q_base_norm.clamp_min(
            1e-8
        )
    )

    return {
        "seed_norm":
            seed_norm,

        "q_base_norm":
            q_base_norm,

        "seeded_q_norm":
            seeded_q_norm,

        "ratio":
            ratio,
    }


# ============================================================
# Seed scale ablation using EXACT SAME selected route
# ============================================================

def get_seed_scale_parameters(
    model,
):
    params = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        main_attention = (
            block.attn.main_attention
        )

        if not hasattr(
            main_attention,
            "seed_scale",
        ):
            raise AttributeError(
                "BoundMainAttention has no attribute 'seed_scale'. "
                f"Block {block_idx} parameters: "
                f"{[n for n, _ in main_attention.named_parameters()]}"
            )

        params.append(
            main_attention.seed_scale
        )

    return params


@torch.no_grad()
def forced_route_forward(
    model,
    images,
    selected_combo_indices,
    combo_table,
):
    forced = []

    combo_device = combo_table.to(
        images.device
    )

    for block_idx in range(
        model.depth
    ):
        forced.append(
            combo_device[
                selected_combo_indices[
                    :,
                    block_idx
                ]
            ]
        )

    return model(
        images,
        return_info=False,
        collect_taylor=False,
        forced_direct_indices_per_block=forced,
        forced_uniform_mix=True,
    )


# ============================================================
# Stats accumulator
# ============================================================

class RunningTensorStats:
    def __init__(self):
        self.values = []

    def add(self, x):
        self.values.append(
            x.detach().float().cpu().reshape(-1)
        )

    def tensor(self):
        if not self.values:
            return torch.empty(0)

        return torch.cat(
            self.values,
            dim=0,
        )

    def summary(self):
        x = self.tensor()

        if x.numel() == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }

        q = torch.quantile(
            x,
            torch.tensor(
                [
                    0.25,
                    0.50,
                    0.75,
                ]
            ),
        )

        return {
            "mean":
                x.mean().item(),

            "median":
                q[1].item(),

            "q25":
                q[0].item(),

            "q75":
                q[2].item(),
        }


# ============================================================
# Main diagnostic pass
# ============================================================

@torch.no_grad()
def run_diagnostics(
    model,
    predictors,
    loader,
    combo_table,
    device,
    use_amp,
    scale_multipliers,
):
    model.eval()

    for predictor in predictors:
        predictor.eval()

    # -----------------------------
    # Mini diversity
    # block -> metric -> pair -> stats
    # -----------------------------

    similarity_stats = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                RunningTensorStats
            )
        )
    )

    # -----------------------------
    # Seed strength
    # block -> main head -> stats
    # -----------------------------

    seed_ratio_stats = defaultdict(
        lambda: defaultdict(
            RunningTensorStats
        )
    )

    seed_norm_stats = defaultdict(
        lambda: defaultdict(
            RunningTensorStats
        )
    )

    q_norm_stats = defaultdict(
        lambda: defaultdict(
            RunningTensorStats
        )
    )

    # Baseline dynamic route/classification
    baseline_loss_sum = 0.0
    baseline_correct = 0
    total_samples = 0

    # Same-route seed multiplier sweep
    sweep = {
        multiplier: {
            "loss_sum": 0.0,
            "correct": 0,
        }
        for multiplier in scale_multipliers
    }

    seed_params = get_seed_scale_parameters(
        model
    )

    original_seed_scales = [
        p.detach().clone()
        for p in seed_params
    ]

    try:
        for batch_idx, (
            images,
            labels,
        ) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            B = labels.shape[0]

            # Ensure baseline scale before choosing route / collecting info.
            for parameter, original in zip(
                seed_params,
                original_seed_scales,
            ):
                parameter.data.copy_(
                    original
                )

            with (
                torch.amp.autocast(
                    device_type="cuda",
                    enabled=True,
                )
                if (
                    use_amp
                    and
                    device.type == "cuda"
                )
                else nullcontext()
            ):
                (
                    logits,
                    selected,
                    info_list,
                ) = dynamic_forward_with_info(
                    model=model,
                    predictors=predictors,
                    images=images,
                    combo_table=combo_table,
                )

            per_sample_loss = F.cross_entropy(
                logits.float(),
                labels,
                reduction="none",
            )

            baseline_loss_sum += (
                per_sample_loss.sum().item()
            )

            baseline_correct += (
                logits.argmax(
                    dim=-1
                )
                ==
                labels
            ).sum().item()

            total_samples += B

            # -----------------------------------------------
            # Structural diagnostics
            # -----------------------------------------------

            for block_idx, info in enumerate(
                info_list
            ):
                similarity = collect_mini_similarity(
                    info
                )

                for metric_name, pair_values in similarity.items():
                    for pair, values in pair_values.items():
                        similarity_stats[
                            block_idx
                        ][
                            metric_name
                        ][
                            pair
                        ].add(
                            values
                        )

                seed_strength = collect_seed_strength(
                    info
                )

                Hmain = (
                    seed_strength[
                        "ratio"
                    ].shape[1]
                )

                for main_head in range(
                    Hmain
                ):
                    seed_ratio_stats[
                        block_idx
                    ][
                        main_head
                    ].add(
                        seed_strength[
                            "ratio"
                        ][
                            :,
                            main_head
                        ]
                    )

                    seed_norm_stats[
                        block_idx
                    ][
                        main_head
                    ].add(
                        seed_strength[
                            "seed_norm"
                        ][
                            :,
                            main_head
                        ]
                    )

                    q_norm_stats[
                        block_idx
                    ][
                        main_head
                    ].add(
                        seed_strength[
                            "q_base_norm"
                        ][
                            :,
                            main_head
                        ]
                    )

            # -----------------------------------------------
            # SAME route, only change seed strength.
            # -----------------------------------------------

            for multiplier in scale_multipliers:
                for parameter, original in zip(
                    seed_params,
                    original_seed_scales,
                ):
                    parameter.data.copy_(
                        original
                        *
                        float(
                            multiplier
                        )
                    )

                with (
                    torch.amp.autocast(
                        device_type="cuda",
                        enabled=True,
                    )
                    if (
                        use_amp
                        and
                        device.type == "cuda"
                    )
                    else nullcontext()
                ):
                    scaled_logits = forced_route_forward(
                        model=model,
                        images=images,
                        selected_combo_indices=selected,
                        combo_table=combo_table,
                    )

                scaled_loss = F.cross_entropy(
                    scaled_logits.float(),
                    labels,
                    reduction="sum",
                )

                sweep[
                    multiplier
                ][
                    "loss_sum"
                ] += scaled_loss.item()

                sweep[
                    multiplier
                ][
                    "correct"
                ] += (
                    scaled_logits.argmax(
                        dim=-1
                    )
                    ==
                    labels
                ).sum().item()

            print(
                f"Diagnostic: "
                f"{total_samples}/"
                f"{len(loader.dataset)}"
            )

    finally:
        # Always restore trained seed scales.
        for parameter, original in zip(
            seed_params,
            original_seed_scales,
        ):
            parameter.data.copy_(
                original
            )

    return {
        "baseline_ce":
            baseline_loss_sum
            /
            total_samples,

        "baseline_accuracy":
            100.0
            *
            baseline_correct
            /
            total_samples,

        "similarity":
            similarity_stats,

        "seed_ratio":
            seed_ratio_stats,

        "seed_norm":
            seed_norm_stats,

        "q_norm":
            q_norm_stats,

        "sweep": {
            multiplier: {
                "ce":
                    values[
                        "loss_sum"
                    ]
                    /
                    total_samples,

                "accuracy":
                    100.0
                    *
                    values[
                        "correct"
                    ]
                    /
                    total_samples,
            }
            for multiplier, values in sweep.items()
        },
    }


# ============================================================
# Reporting
# ============================================================

def format_summary(stats):
    s = stats.summary()

    return (
        f"mean={s['mean']:.4f}, "
        f"median={s['median']:.4f}, "
        f"Q25={s['q25']:.4f}, "
        f"Q75={s['q75']:.4f}"
    )


def print_similarity_report(
    similarity_stats,
    depth,
):
    print(
        "\n================ MINI HEAD DIVERSITY ================"
    )

    print(
        "Cosine similarity: 1에 가까울수록 두 Mini Head 정보가 비슷함."
    )

    for block_idx in range(
        depth
    ):
        print(
            f"\nBlock {block_idx}"
        )

        for metric_name, korean_name in [
            ("cls", "CLS context"),
            ("patch", "Patch-mean context"),
            ("full", "Full context"),
        ]:
            pair_map = similarity_stats[
                block_idx
            ][
                metric_name
            ]

            all_values = []

            print(
                f"\n  {korean_name}:"
            )

            for pair in sorted(
                pair_map.keys()
            ):
                values = pair_map[
                    pair
                ].tensor()

                all_values.append(
                    values
                )

                print(
                    f"    H{pair[0]}-H{pair[1]}: "
                    f"{format_summary(pair_map[pair])}"
                )

            if all_values:
                merged = RunningTensorStats()
                merged.add(
                    torch.cat(
                        all_values
                    )
                )

                print(
                    "    전체 pair: "
                    f"{format_summary(merged)}"
                )


def print_seed_strength_report(
    result,
    depth,
    main_heads,
):
    print(
        "\n================ DIRECT SEED STRENGTH ================"
    )

    print(
        "비율 = ||Mini seed가 Q에 더한 값|| / ||원래 Main Q||"
    )

    for block_idx in range(
        depth
    ):
        print(
            f"\nBlock {block_idx}:"
        )

        merged_ratio = RunningTensorStats()

        for main_head in range(
            main_heads
        ):
            ratio = result[
                "seed_ratio"
            ][
                block_idx
            ][
                main_head
            ]

            merged_ratio.add(
                ratio.tensor()
            )

            seed_s = result[
                "seed_norm"
            ][
                block_idx
            ][
                main_head
            ].summary()

            q_s = result[
                "q_norm"
            ][
                block_idx
            ][
                main_head
            ].summary()

            print(
                f"  Main H{main_head}: "
                f"ratio {format_summary(ratio)} | "
                f"seed norm mean={seed_s['mean']:.4f} | "
                f"Q_base norm mean={q_s['mean']:.4f}"
            )

        print(
            "  전체 Main head ratio: "
            f"{format_summary(merged_ratio)}"
        )


def print_scale_sweep(
    result,
):
    print(
        "\n================ SEED 영향 ABLATION ================"
    )

    print(
        "같은 Direct 경로를 강제로 유지하고 Mini seed 세기만 변경."
    )

    baseline_ce = result[
        "baseline_ce"
    ]

    baseline_acc = result[
        "baseline_accuracy"
    ]

    print(
        f"\n동적 원본 forward: "
        f"CE={baseline_ce:.6f}, "
        f"Acc={baseline_acc:.2f}%"
    )

    base = result[
        "sweep"
    ][
        1.0
    ]

    print(
        "\nMultiplier | CE | Accuracy | ΔCE vs x1 | ΔAcc vs x1"
    )

    for multiplier in sorted(
        result[
            "sweep"
        ].keys()
    ):
        m = result[
            "sweep"
        ][
            multiplier
        ]

        print(
            f"{multiplier:>10.2f} | "
            f"{m['ce']:.6f} | "
            f"{m['accuracy']:.2f}% | "
            f"{m['ce'] - base['ce']:+.6f} | "
            f"{m['accuracy'] - base['accuracy']:+.2f}%p"
        )


def print_interpretation_hints(
    result,
    depth,
    main_heads,
):
    # Overall seed ratio.
    ratio_values = []

    for b in range(depth):
        for h in range(main_heads):
            ratio_values.append(
                result[
                    "seed_ratio"
                ][b][h].tensor()
            )

    overall_ratio = torch.cat(
        ratio_values
    ).mean().item()

    # Overall full-context cosine similarity.
    similarity_values = []

    for b in range(depth):
        pair_map = result[
            "similarity"
        ][b]["full"]

        for stats in pair_map.values():
            similarity_values.append(
                stats.tensor()
            )

    overall_similarity = torch.cat(
        similarity_values
    ).mean().item()

    seed_off = result[
        "sweep"
    ].get(
        0.0,
        None,
    )

    seed_on = result[
        "sweep"
    ][
        1.0
    ]

    print(
        "\n================ 빠른 판정 기준 ================"
    )

    print(
        f"전체 평균 seed/Q 비율: "
        f"{overall_ratio:.4f}"
    )

    print(
        f"전체 평균 Mini Head full-context cosine: "
        f"{overall_similarity:.4f}"
    )

    if seed_off is not None:
        print(
            f"Seed OFF → ON(x1) CE 변화: "
            f"{seed_on['ce'] - seed_off['ce']:+.6f}"
        )

        print(
            f"Seed OFF → ON(x1) 정확도 변화: "
            f"{seed_on['accuracy'] - seed_off['accuracy']:+.2f}%p"
        )

    print(
        "\n해석:"
    )

    print(
        "- seed/Q 비율이 매우 작고 x0↔x1 성능 차이도 작으면 "
        "Mini→Main 전달 세기가 병목일 가능성이 큼."
    )

    print(
        "- Mini Head cosine이 매우 높으면 Head들이 서로 비슷한 정보를 만들어 "
        "어떤 조합을 골라도 차이가 작을 가능성이 큼."
    )

    print(
        "- seed 영향도 있고 Mini 다양성도 충분한데 routing 차이가 작다면 "
        "Balanced warm-up/학습 목적을 다시 봐야 함."
    )

    print(
        "- x2/x4에서 CE가 좋아지면 다음 실험은 seed 주입 세기/정규화 재설계가 유력."
    )

    print(
        "- x2/x4에서 오히려 나빠지면 단순 seed 증폭은 해결책이 아님."
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
        "\n주의: 이 실험은 official CIFAR-10 test를 사용하지 않습니다."
    )

    print(
        "기존에 쓰지 않은 CIFAR-10 train split의 별도 진단 샘플을 사용합니다."
    )

    stage1 = load_checkpoint(
        args.stage1_checkpoint,
        device,
        "Stage-1 checkpoint",
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

    predictor_checkpoint = load_checkpoint(
        args.predictor_checkpoint,
        device,
        "Refined predictor checkpoint",
    )

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
        predictor_checkpoint,
        model,
        feature_dim,
        device,
    )

    combinations = list(
        itertools.combinations(
            range(
                model.mini_heads
            ),
            model.direct_k,
        )
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    dataset = build_diagnostic_dataset(
        args
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    multipliers = parse_float_list(
        args.seed_scale_multipliers
    )

    print(
        "\nDiagnostic samples:",
        len(dataset),
    )

    print(
        "Seed scale multipliers:",
        multipliers,
    )

    # Show learned seed-scale parameters themselves.
    print(
        "\nLearned seed_scale parameters:"
    )

    for block_idx, block in enumerate(
        model.blocks
    ):
        scale = (
            block.attn.main_attention
            .seed_scale
            .detach()
            .cpu()
        )

        print(
            f"Block {block_idx}: "
            f"{scale.tolist()}"
        )

    result = run_diagnostics(
        model=model,
        predictors=predictors,
        loader=loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
        scale_multipliers=multipliers,
    )

    print_similarity_report(
        result[
            "similarity"
        ],
        model.depth,
    )

    print_seed_strength_report(
        result,
        model.depth,
        model.main_heads,
    )

    print_scale_sweep(
        result
    )

    print_interpretation_hints(
        result,
        model.depth,
        model.main_heads,
    )

    print(
        "\nDirect-path diagnostic completed."
    )


if __name__ == "__main__":
    main()
