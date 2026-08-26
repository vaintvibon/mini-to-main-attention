import argparse
import itertools
import math
import os
import random
from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.utility_interaction_predictor import UtilityInteractionPredictor
from models.contextual_utility_interaction_predictor import (
    ContextualUtilityInteractionPredictor,
)


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
        "--backbone-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage1_cifar10_seedscale_tuned.pt"
        ),
    )

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
            "stage3_block0_context_probe.pt"
        ),
    )

    p.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Same Stage-3 regions already used before.
    p.add_argument(
        "--stage1-train-subset",
        type=int,
        default=4096,
    )
    p.add_argument(
        "--stage1-val-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--utility-train-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--utility-val-subset",
        type=int,
        default=500,
    )
    p.add_argument(
        "--diagnostic-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--scale-train-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--scale-val-subset",
        type=int,
        default=500,
    )
    p.add_argument(
        "--previous-heldout-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--decision-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--global-train-subset",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--global-val-subset",
        type=int,
        default=500,
    )

    p.add_argument(
        "--feature-batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    p.add_argument(
        "--inner-val-samples",
        type=int,
        default=200,
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

    p.add_argument(
        "--amp",
        action="store_true",
    )

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Load backbone / route cache
# ============================================================

def load_file(
    path,
    device,
    name,
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def build_model_from_config(
    config,
):
    return DynamicMiniMainViT(
        img_size=32,
        patch_size=4,
        num_classes=10,

        embed_dim=config.get(
            "embed_dim",
            192,
        ),
        depth=config.get(
            "depth",
            2,
        ),

        main_heads=config.get(
            "main_heads",
            3,
        ),

        mini_heads=config.get(
            "mini_heads",
            4,
        ),
        mini_head_dim=config.get(
            "mini_head_dim",
            16,
        ),
        pool_ratio=2,

        utility_hidden_dim=64,

        direct_k=config.get(
            "direct_k",
            2,
        ),
        mix_temperature=1.0,

        bind_dim=64,
        bind_temperature=1.0,

        mlp_ratio=4.0,

        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


# ============================================================
# Data
# ============================================================

def get_transform():
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

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )


def build_stage3_sets(
    args,
):
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

    start = (
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

    train_end = (
        start
        +
        args.global_train_subset
    )

    val_end = (
        train_end
        +
        args.global_val_subset
    )

    print(
        "\nFeature-probe data:"
    )

    print(
        f"  Stage-3 train: "
        f"[{start}, {train_end})"
    )

    print(
        f"  Stage-3 val:   "
        f"[{train_end}, {val_end})"
    )

    print(
        "  These data were already used in Stage 3."
    )

    print(
        "  No fresh held-out or official CIFAR-10 test is consumed."
    )

    return (
        Subset(
            base,
            permutation[
                start:
                train_end
            ],
        ),
        Subset(
            base,
            permutation[
                train_end:
                val_end
            ],
        ),
    )


# ============================================================
# Block-0 rich feature extraction
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


def extract_original_features(
    mini_contexts,
    mini_attn,
    eps=1e-6,
):
    B, H, N, Dh = (
        mini_contexts.shape
    )

    cls_context = mini_contexts[
        :,
        :,
        0,
        :
    ]

    if N > 1:
        patch_mean = (
            mini_contexts[
                :,
                :,
                1:,
                :
            ].mean(
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


def safe_cosine(
    a,
    b,
    dim=-1,
    eps=1e-8,
):
    return F.cosine_similarity(
        a,
        b,
        dim=dim,
        eps=eps,
    )


@torch.no_grad()
def extract_rich_batch(
    model,
    images,
    pair_indices,
):
    """
    Returns:
      original_head_features [B,4,34]

      rich_head_features [B,4,37]
        original 34
        + CLS-context norm
        + patch-mean norm
        + mean token-context norm

      global_features [B,384]
        normalized current block input:
          CLS token [192]
          mean patch token [192]

      pair_features [B,6,5]
        CLS-context cosine
        patch-mean cosine
        full Mini-context cosine
        Mini-attention cosine
        absolute mean-context-norm difference
    """

    x = prepare_tokens(
        model,
        images,
    )

    block0 = model.blocks[0]

    x_norm = block0.norm1(
        x
    )

    (
        mini_contexts,
        mini_attn,
    ) = block0.attn.mini_attention(
        x_norm,
        patch_hw=model.patch_hw,
    )

    original = extract_original_features(
        mini_contexts,
        mini_attn,
    )

    cls_context = mini_contexts[
        :,
        :,
        0,
        :
    ]

    if mini_contexts.shape[2] > 1:
        patch_mean = (
            mini_contexts[
                :,
                :,
                1:,
                :
            ].mean(
                dim=2
            )
        )
    else:
        patch_mean = cls_context

    cls_norm = cls_context.norm(
        dim=-1,
    )

    patch_norm = patch_mean.norm(
        dim=-1,
    )

    mean_token_norm = (
        mini_contexts.norm(
            dim=-1,
        ).mean(
            dim=-1,
        )
    )

    rich_head = torch.cat(
        [
            original,
            cls_norm[
                ...,
                None,
            ],
            patch_norm[
                ...,
                None,
            ],
            mean_token_norm[
                ...,
                None,
            ],
        ],
        dim=-1,
    )

    if x_norm.shape[1] > 1:
        x_patch_mean = (
            x_norm[
                :,
                1:,
                :
            ].mean(
                dim=1
            )
        )
    else:
        x_patch_mean = (
            x_norm[
                :,
                0,
                :
            ]
        )

    global_features = torch.cat(
        [
            x_norm[
                :,
                0,
                :
            ],
            x_patch_mean,
        ],
        dim=-1,
    )

    flat_context = mini_contexts.flatten(
        start_dim=2
    )

    flat_attn = mini_attn.flatten(
        start_dim=2
    )

    pair_features = []

    for i, j in pair_indices:
        i = int(i)
        j = int(j)

        cls_cos = safe_cosine(
            cls_context[
                :,
                i,
                :,
            ],
            cls_context[
                :,
                j,
                :,
            ],
        )

        patch_cos = safe_cosine(
            patch_mean[
                :,
                i,
                :,
            ],
            patch_mean[
                :,
                j,
                :,
            ],
        )

        full_cos = safe_cosine(
            flat_context[
                :,
                i,
                :,
            ],
            flat_context[
                :,
                j,
                :,
            ],
        )

        attn_cos = safe_cosine(
            flat_attn[
                :,
                i,
                :,
            ],
            flat_attn[
                :,
                j,
                :,
            ],
        )

        norm_diff = (
            mean_token_norm[
                :,
                i,
            ]
            -
            mean_token_norm[
                :,
                j,
            ]
        ).abs()

        pair_features.append(
            torch.stack(
                [
                    cls_cos,
                    patch_cos,
                    full_cos,
                    attn_cos,
                    norm_diff,
                ],
                dim=-1,
            )
        )

    pair_features = torch.stack(
        pair_features,
        dim=1,
    )

    return (
        original.float(),
        rich_head.float(),
        global_features.float(),
        pair_features.float(),
    )


@torch.no_grad()
def extract_split_features(
    model,
    dataset,
    pair_indices,
    device,
    batch_size,
    num_workers,
    use_amp,
    split_name,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        drop_last=False,
    )

    original_all = []
    rich_head_all = []
    global_all = []
    pair_all = []

    seen = 0

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device,
            use_amp,
        ):
            (
                original,
                rich_head,
                global_features,
                pair_features,
            ) = extract_rich_batch(
                model,
                images,
                pair_indices,
            )

        original_all.append(
            original.cpu()
        )

        rich_head_all.append(
            rich_head.cpu()
        )

        global_all.append(
            global_features.cpu()
        )

        pair_all.append(
            pair_features.cpu()
        )

        seen += images.shape[0]

        print(
            f"{split_name}: "
            f"{seen}/{len(dataset)}"
        )

    return {
        "original_head_features":
            torch.cat(
                original_all,
                dim=0,
            ),

        "rich_head_features":
            torch.cat(
                rich_head_all,
                dim=0,
            ),

        "global_features":
            torch.cat(
                global_all,
                dim=0,
            ),

        "pair_features":
            torch.cat(
                pair_all,
                dim=0,
            ),
    }


def load_or_build_feature_cache(
    model,
    train_set,
    val_set,
    pair_indices,
    device,
    args,
):
    if (
        os.path.exists(
            args.feature_cache
        )
        and
        not args.rebuild_feature_cache
    ):
        print(
            "\nLoading rich feature cache:"
        )

        print(
            args.feature_cache
        )

        cache = torch.load(
            args.feature_cache,
            map_location="cpu",
            weights_only=False,
        )

        return (
            cache["train"],
            cache["val"],
        )

    print(
        "\nGenerating Block-0 rich features..."
    )

    train_features = extract_split_features(
        model=model,
        dataset=train_set,
        pair_indices=pair_indices,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        use_amp=(
            args.amp
            and
            device.type
            ==
            "cuda"
        ),
        split_name="train",
    )

    val_features = extract_split_features(
        model=model,
        dataset=val_set,
        pair_indices=pair_indices,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        use_amp=(
            args.amp
            and
            device.type
            ==
            "cuda"
        ),
        split_name="val",
    )

    os.makedirs(
        os.path.dirname(
            args.feature_cache
        )
        or
        ".",
        exist_ok=True,
    )

    torch.save(
        {
            "train":
                train_features,

            "val":
                val_features,

            "pair_indices":
                pair_indices,
        },
        args.feature_cache,
    )

    print(
        "\nSaved rich feature cache:"
    )

    print(
        args.feature_cache
    )

    return (
        train_features,
        val_features,
    )


# ============================================================
# Oracle continuation targets for Block 0
# ============================================================

def get_block0_oracle_costs(
    route_cache_split,
):
    route_losses = route_cache_split[
        "route_losses"
    ].float()

    # cost[p0] = best final CE attainable after choosing p0.
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

class ProbeDataset(Dataset):
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

        self.pair_costs = pair_costs[
            indices
        ].float()

        self.utility_target = utility_target[
            indices
        ].float()

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


# ============================================================
# Training
# ============================================================

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
        "baseline64",
        "baseline128",
    ):
        return model(
            original_head,
            return_info=True,
        )

    if variant == "contextual":
        return model(
            rich_head,
            global_features,
            pair_features,
            return_info=True,
        )

    raise ValueError(
        variant
    )


def compute_loss(
    model,
    variant,
    batch,
    args,
    eps=1e-8,
):
    (
        original_head,
        rich_head,
        global_features,
        pair_features,
        pair_costs,
        utility_target,
    ) = batch

    pair_scores, info = forward_variant(
        model,
        variant,
        batch,
    )

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

    normalized_cost = (
        pair_costs
        -
        min_cost
    ) / spread

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

    return total


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

        pair_scores, _ = forward_variant(
            model,
            variant,
            batch,
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


def parameter_count(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def build_probe_model(
    variant,
    original_feature_dim,
    rich_feature_dim,
    global_feature_dim,
    pair_feature_dim,
    mini_heads,
    device,
):
    if variant == "baseline64":
        model = UtilityInteractionPredictor(
            feature_dim=original_feature_dim,
            mini_heads=mini_heads,
            direct_k=2,
            hidden_dim=64,
            dropout=0.0,
        )

    elif variant == "baseline128":
        model = UtilityInteractionPredictor(
            feature_dim=original_feature_dim,
            mini_heads=mini_heads,
            direct_k=2,
            hidden_dim=128,
            dropout=0.0,
        )

    elif variant == "contextual":
        model = ContextualUtilityInteractionPredictor(
            head_feature_dim=rich_feature_dim,
            global_feature_dim=global_feature_dim,
            pair_feature_dim=pair_feature_dim,
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


def train_one_repeat(
    variant,
    repeat_seed,
    train_features,
    val_features,
    train_costs,
    val_costs,
    combo_table,
    mini_heads,
    device,
    args,
):
    seed_everything(
        repeat_seed
    )

    n_train_total = (
        train_costs.shape[0]
    )

    if (
        args.inner_val_samples
        <=
        0
        or
        args.inner_val_samples
        >=
        n_train_total
    ):
        raise ValueError(
            "--inner-val-samples must be between 1 and Stage-3 train size-1."
        )

    g = torch.Generator().manual_seed(
        repeat_seed
    )

    perm = torch.randperm(
        n_train_total,
        generator=g,
    )

    inner_val_idx = perm[
        :
        args.inner_val_samples
    ]

    inner_train_idx = perm[
        args.inner_val_samples:
    ]

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

    inner_train_dataset = ProbeDataset(
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

    inner_val_dataset = ProbeDataset(
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

    external_val_idx = torch.arange(
        val_costs.shape[0]
    )

    external_val_dataset = ProbeDataset(
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
        inner_train_dataset,
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
        inner_val_dataset,
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
        external_val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    model = build_probe_model(
        variant=variant,
        original_feature_dim=train_features[
            "original_head_features"
        ].shape[-1],
        rich_feature_dim=train_features[
            "rich_head_features"
        ].shape[-1],
        global_feature_dim=train_features[
            "global_features"
        ].shape[-1],
        pair_feature_dim=train_features[
            "pair_features"
        ].shape[-1],
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

        loss_sum = 0.0
        seen = 0

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

            B = batch[
                4
            ].shape[0]

            loss_sum += (
                loss.item()
                *
                B
            )

            seen += B

        scheduler.step()

        inner_metrics = evaluate(
            model,
            variant,
            inner_val_loader,
            device,
        )

        if (
            inner_metrics[
                "mean_regret"
            ]
            <
            best_inner_regret
        ):
            best_inner_regret = (
                inner_metrics[
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
    ] = parameter_count(
        model
    )

    external[
        "state_dict"
    ] = deepcopy(
        model.state_dict()
    )

    return external


# ============================================================
# Static Block-0 baseline for the same oracle-continuation target
# ============================================================

def static_p0_baseline(
    train_costs,
    val_costs,
):
    best_train_p0 = int(
        train_costs.mean(
            dim=0
        ).argmin().item()
    )

    selected = val_costs[
        :,
        best_train_p0
    ]

    oracle = val_costs.min(
        dim=-1
    ).values

    regret = (
        selected
        -
        oracle
    )

    return {
        "p0":
            best_train_p0,

        "selected_ce":
            selected.mean().item(),

        "oracle_ce":
            oracle.mean().item(),

        "mean_regret":
            regret.mean().item(),
    }


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


def summarize_variant(
    name,
    results,
):
    exact_mean, exact_std = mean_std(
        [
            r["exact"]
            for r in results
        ]
    )

    regret_mean, regret_std = mean_std(
        [
            r["mean_regret"]
            for r in results
        ]
    )

    ce_mean, ce_std = mean_std(
        [
            r["selected_ce"]
            for r in results
        ]
    )

    epoch_mean, epoch_std = mean_std(
        [
            r["best_epoch"]
            for r in results
        ]
    )

    print(
        f"\n{name}"
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
        "exact_mean":
            exact_mean,

        "regret_mean":
            regret_mean,

        "ce_mean":
            ce_mean,
    }


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
        "Test whether richer Block-0 information can predict ORACLE continuation value "
        "better than the current 34-D Mini-head features."
    )

    print(
        "This is a feature-sufficiency diagnostic, not a final routing evaluation."
    )

    backbone = load_file(
        args.backbone_checkpoint,
        device,
        "Backbone checkpoint",
    )

    model = build_model_from_config(
        backbone[
            "config"
        ]
    ).to(
        device
    )

    model.load_state_dict(
        backbone[
            "model"
        ],
        strict=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(
            False
        )

    if model.depth != 2:
        raise ValueError(
            "Current probe expects depth=2."
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

    print(
        "\nDirect pairs:"
    )

    print(
        combo_table
    )

    route_cache = load_file(
        args.route_cache,
        "cpu",
        "Stage-3 36-route cache",
    )

    train_route = route_cache[
        "train"
    ]

    val_route = route_cache[
        "val"
    ]

    train_set, val_set = (
        build_stage3_sets(
            args
        )
    )

    train_features, val_features = (
        load_or_build_feature_cache(
            model=model,
            train_set=train_set,
            val_set=val_set,
            pair_indices=combinations,
            device=device,
            args=args,
        )
    )

    if (
        train_features[
            "original_head_features"
        ].shape[0]
        !=
        train_route[
            "route_losses"
        ].shape[0]
    ):
        raise RuntimeError(
            "Train feature cache and route cache sample counts do not match."
        )

    if (
        val_features[
            "original_head_features"
        ].shape[0]
        !=
        val_route[
            "route_losses"
        ].shape[0]
    ):
        raise RuntimeError(
            "Val feature cache and route cache sample counts do not match."
        )

    print(
        "\nFeature shapes:"
    )

    for key, value in train_features.items():
        print(
            f"  {key}: "
            f"{tuple(value.shape)}"
        )

    train_costs = (
        get_block0_oracle_costs(
            train_route
        )
    )

    val_costs = (
        get_block0_oracle_costs(
            val_route
        )
    )

    print(
        "\nTarget:"
    )

    print(
        "  cost[p0] = min over all 6 Block-1 choices of final classification CE."
    )

    print(
        f"  train costs: "
        f"{tuple(train_costs.shape)}"
    )

    print(
        f"  val costs: "
        f"{tuple(val_costs.shape)}"
    )

    static_result = (
        static_p0_baseline(
            train_costs,
            val_costs,
        )
    )

    print(
        "\n================ STATIC BLOCK-0 BASELINE ================"
    )

    print(
        f"Best fixed Block-0 pair from Stage-3 train: "
        f"{combinations[static_result['p0']]}"
    )

    print(
        f"Val selected continuation CE: "
        f"{static_result['selected_ce']:.6f}"
    )

    print(
        f"Val oracle continuation CE: "
        f"{static_result['oracle_ce']:.6f}"
    )

    print(
        f"Val mean regret: "
        f"{static_result['mean_regret']:.8e}"
    )

    variants = [
        "baseline64",
        "baseline128",
        "contextual",
    ]

    all_results = {
        variant: []
        for variant in variants
    }

    best_context_state = None
    best_context_regret = float(
        "inf"
    )

    print(
        "\n================ FEATURE PROBE TRAINING ================"
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
            result = train_one_repeat(
                variant=variant,
                repeat_seed=repeat_seed,
                train_features=train_features,
                val_features=val_features,
                train_costs=train_costs,
                val_costs=val_costs,
                combo_table=combo_table,
                mini_heads=model.mini_heads,
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
                f"best_epoch={result['best_epoch']}"
            )

            if (
                variant
                ==
                "contextual"
                and
                result[
                    "mean_regret"
                ]
                <
                best_context_regret
            ):
                best_context_regret = (
                    result[
                        "mean_regret"
                    ]
                )

                best_context_state = (
                    result[
                        "state_dict"
                    ]
                )

    print(
        "\n================ PROBE SUMMARY ================"
    )

    summaries = {}

    summaries[
        "baseline64"
    ] = summarize_variant(
        "A. Current 34-D features / hidden 64",
        all_results[
            "baseline64"
        ],
    )

    summaries[
        "baseline128"
    ] = summarize_variant(
        "B. Current 34-D features / hidden 128 (capacity control)",
        all_results[
            "baseline128"
        ],
    )

    summaries[
        "contextual"
    ] = summarize_variant(
        "C. Rich Block-0 contextual features / hidden 64",
        all_results[
            "contextual"
        ],
    )

    base_regret = summaries[
        "baseline64"
    ][
        "regret_mean"
    ]

    wide_regret = summaries[
        "baseline128"
    ][
        "regret_mean"
    ]

    rich_regret = summaries[
        "contextual"
    ][
        "regret_mean"
    ]

    print(
        "\n================ FEATURE VALUE ================"
    )

    print(
        f"Static Block-0 regret: "
        f"{static_result['mean_regret']:.8e}"
    )

    print(
        f"34-D baseline regret: "
        f"{base_regret:.8e}"
    )

    print(
        f"34-D wider-model regret: "
        f"{wide_regret:.8e}"
    )

    print(
        f"Rich contextual regret: "
        f"{rich_regret:.8e}"
    )

    if base_regret > 0:
        print(
            f"Rich vs 34-D regret reduction: "
            f"{100.0 * (base_regret - rich_regret) / base_regret:+.2f}%"
        )

    if wide_regret > 0:
        print(
            f"Rich vs wider 34-D regret reduction: "
            f"{100.0 * (wide_regret - rich_regret) / wide_regret:+.2f}%"
        )

    # Save best contextual probe for inspection only.
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
            "contextual_predictor":
                best_context_state,

            "config": {
                "head_feature_dim":
                    train_features[
                        "rich_head_features"
                    ].shape[-1],

                "global_feature_dim":
                    train_features[
                        "global_features"
                    ].shape[-1],

                "pair_feature_dim":
                    train_features[
                        "pair_features"
                    ].shape[-1],

                "mini_heads":
                    model.mini_heads,

                "direct_k":
                    model.direct_k,

                "hidden_dim":
                    64,

                "target":
                    "block0_oracle_continuation_value",
            },

            "probe_summary":
                summaries,

            "static_result":
                static_result,

            "source_backbone":
                args.backbone_checkpoint,

            "source_route_cache":
                args.route_cache,
        },
        args.output_checkpoint,
    )

    print(
        "\nSaved best contextual diagnostic predictor:"
    )

    print(
        args.output_checkpoint
    )

    print(
        "\n================ FEATURE-PROBE VERDICT ================"
    )

    rich_beats_base = (
        rich_regret
        <
        0.90
        *
        base_regret
    )

    rich_beats_wide = (
        rich_regret
        <
        0.95
        *
        wide_regret
    )

    if (
        rich_beats_base
        and
        rich_beats_wide
    ):
        print(
            "FEATURE BOTTLENECK SUPPORTED: richer Block-0 state/redundancy information "
            "substantially improves oracle-continuation prediction."
        )

        print(
            "Next step: integrate the contextual feature path into the real sequential router, "
            "then validate on a new untouched split."
        )

    elif (
        rich_regret
        <
        base_regret
    ):
        print(
            "WEAK FEATURE SIGNAL: richer features help somewhat, "
            "but the gain is not yet large enough to call feature insufficiency the main bottleneck."
        )

        print(
            "Inspect which feature group contributes before modifying the production router."
        )

    else:
        print(
            "FEATURE HYPOTHESIS NOT SUPPORTED: the richer feature set does not improve "
            "Block-0 oracle-continuation prediction."
        )

        print(
            "Do not enlarge the production router yet; the bottleneck is likely target noise, "
            "model formulation, or a deeper limitation of predicting route value from pre-route state."
        )


if __name__ == "__main__":
    main()
