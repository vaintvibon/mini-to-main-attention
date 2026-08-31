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
        "--init-predictor-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument(
        "--output-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage3_global_value_predictor.pt"
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

    p.add_argument("--rebuild-cache", action="store_true")

    p.add_argument("--seed", type=int, default=42)

    # Every region consumed before this stage.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)
    p.add_argument("--diagnostic-subset", type=int, default=1000)
    p.add_argument("--scale-train-subset", type=int, default=1000)
    p.add_argument("--scale-val-subset", type=int, default=500)
    p.add_argument("--previous-heldout-subset", type=int, default=1000)
    p.add_argument("--decision-subset", type=int, default=1000)

    # Fresh data for global-value training.
    p.add_argument("--global-train-subset", type=int, default=1000)
    p.add_argument("--global-val-subset", type=int, default=500)

    p.add_argument("--teacher-batch-size", type=int, default=64)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--block1-epochs", type=int, default=30)
    p.add_argument("--block0-epochs", type=int, default=30)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--regret-weight", type=float, default=1.0)
    p.add_argument("--pair-kl-weight", type=float, default=0.5)
    p.add_argument("--utility-kl-weight", type=float, default=0.5)
    p.add_argument("--interaction-l2-weight", type=float, default=0.01)
    p.add_argument("--pair-temperature", type=float, default=0.5)

    p.add_argument("--amp", action="store_true")

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


def load_predictors(
    checkpoint,
    model,
    feature_dim,
    device,
):
    config = checkpoint.get("config", {})

    hidden_dim = int(
        config.get("hidden_dim", 64)
    )

    dropout = float(
        config.get("dropout", 0.0)
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

    states = checkpoint["predictors"]

    for block_idx, predictor in enumerate(predictors):
        predictor.load_state_dict(
            states[f"block_{block_idx}"],
            strict=True,
        )

    return predictors, hidden_dim, dropout


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


def build_global_value_sets(args):
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

    if val_end > len(base):
        raise ValueError(
            f"Requested split ends at {val_end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    print("\nFresh Stage-3 split:")
    print(f"  train offset: [{start}, {train_end})")
    print(f"  val offset:   [{train_end}, {val_end})")
    print("  official CIFAR-10 test is NOT used.")

    return (
        Subset(
            base,
            permutation[start:train_end],
        ),
        Subset(
            base,
            permutation[train_end:val_end],
        ),
    )


# ============================================================
# Features / route enumeration
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


@torch.no_grad()
def get_block_features(
    model,
    block,
    x,
):
    x_norm = block.norm1(x)

    (
        mini_contexts,
        mini_attn,
    ) = block.attn.mini_attention(
        x_norm,
        patch_hw=model.patch_hw,
    )

    return extract_mini_features(
        mini_contexts,
        mini_attn,
    )


@torch.no_grad()
def generate_global_value_split(
    model,
    dataset,
    combinations,
    device,
    batch_size,
    num_workers,
    use_amp,
    split_name,
):
    """
    Exact depth-2 route table.

    For each image:
      block0_features: [H,F]

      for every p0 in 6 Block-0 pairs:
          execute Block 0 with p0
          cache Block-1 features for that actual state
          evaluate every p1 in 6 Block-1 pairs

      route_losses[p0,p1] = final classification CE
    """

    if model.depth != 2:
        raise ValueError(
            "This exact global-value experiment currently supports depth=2 only."
        )

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

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
        device=device,
    )

    block0 = model.blocks[0]
    block1 = model.blocks[1]

    b0_features_all = []
    b1_features_all = []
    route_losses_all = []

    seen = 0

    print(
        f"\nGenerating exact 36-route teacher: {split_name}"
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

        with amp_context(
            device,
            use_amp,
        ):
            x0 = prepare_tokens(
                model,
                images,
            )

            block0_features = get_block_features(
                model,
                block0,
                x0,
            )

            block1_features_for_p0 = []
            losses_for_p0 = []

            for p0_idx in range(
                len(combinations)
            ):
                p0_pair = (
                    combo_table[
                        p0_idx
                    ][None, :]
                    .expand(B, -1)
                    .clone()
                )

                x1 = block0(
                    x0,
                    patch_hw=model.patch_hw,
                    return_info=False,
                    collect_taylor=False,
                    forced_direct_indices=p0_pair,
                    forced_uniform_mix=True,
                )

                block1_features = (
                    get_block_features(
                        model,
                        block1,
                        x1,
                    )
                )

                block1_features_for_p0.append(
                    block1_features
                )

                p1_losses = []

                for p1_idx in range(
                    len(combinations)
                ):
                    p1_pair = (
                        combo_table[
                            p1_idx
                        ][None, :]
                        .expand(B, -1)
                        .clone()
                    )

                    x2 = block1(
                        x1,
                        patch_hw=model.patch_hw,
                        return_info=False,
                        collect_taylor=False,
                        forced_direct_indices=p1_pair,
                        forced_uniform_mix=True,
                    )

                    logits = model.head(
                        model.norm(x2)[:, 0]
                    )

                    losses = F.cross_entropy(
                        logits.float(),
                        labels,
                        reduction="none",
                    )

                    p1_losses.append(
                        losses
                    )

                losses_for_p0.append(
                    torch.stack(
                        p1_losses,
                        dim=-1,
                    )
                )

        # [B,6,H,F]
        block1_feature_table = torch.stack(
            block1_features_for_p0,
            dim=1,
        )

        # [B,6,6]
        route_loss_table = torch.stack(
            losses_for_p0,
            dim=1,
        )

        b0_features_all.append(
            block0_features.float().cpu()
        )

        b1_features_all.append(
            block1_feature_table.float().cpu()
        )

        route_losses_all.append(
            route_loss_table.float().cpu()
        )

        seen += B

        print(
            f"{split_name}: "
            f"{seen}/{len(dataset)}"
        )

    return {
        "block0_features":
            torch.cat(
                b0_features_all,
                dim=0,
            ),

        "block1_features":
            torch.cat(
                b1_features_all,
                dim=0,
            ),

        "route_losses":
            torch.cat(
                route_losses_all,
                dim=0,
            ),
    }


def load_or_build_cache(
    model,
    train_set,
    val_set,
    combinations,
    device,
    args,
):
    if (
        os.path.exists(args.route_cache)
        and
        not args.rebuild_cache
    ):
        print(
            "\nLoading Stage-3 route cache:"
        )
        print(args.route_cache)

        cache = torch.load(
            args.route_cache,
            map_location="cpu",
            weights_only=False,
        )

        return (
            cache["train"],
            cache["val"],
        )

    train_data = generate_global_value_split(
        model=model,
        dataset=train_set,
        combinations=combinations,
        device=device,
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        use_amp=(
            args.amp
            and
            device.type == "cuda"
        ),
        split_name="train",
    )

    val_data = generate_global_value_split(
        model=model,
        dataset=val_set,
        combinations=combinations,
        device=device,
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        use_amp=(
            args.amp
            and
            device.type == "cuda"
        ),
        split_name="val",
    )

    os.makedirs(
        os.path.dirname(args.route_cache)
        or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "train": train_data,
            "val": val_data,
            "combinations": combinations,
            "backbone_checkpoint":
                args.backbone_checkpoint,
        },
        args.route_cache,
    )

    print(
        "\nSaved Stage-3 route cache:"
    )
    print(args.route_cache)

    return train_data, val_data


# ============================================================
# Utility target derived from pair costs
# ============================================================

def costs_to_utility_target(
    pair_costs,
    combo_table,
    mini_heads,
    eps=1e-8,
):
    """
    Positive utility means including the head tends to LOWER final cost.

    U_h =
        mean(cost | h excluded)
        -
        mean(cost | h included)
    """

    utilities = []

    combo_table = combo_table.cpu()

    for head_idx in range(
        mini_heads
    ):
        included = (
            combo_table == head_idx
        ).any(dim=-1)

        excluded = ~included

        include_cost = (
            pair_costs[
                :,
                included,
            ].mean(dim=-1)
        )

        exclude_cost = (
            pair_costs[
                :,
                excluded,
            ].mean(dim=-1)
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
        std.clamp_min(eps)
    )

    target = torch.softmax(
        normalized,
        dim=-1,
    )

    zero_signal = (
        std.squeeze(-1)
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

    return utility, target


# ============================================================
# Offline datasets
# ============================================================

class PairCostDataset(Dataset):
    def __init__(
        self,
        features,
        pair_costs,
        utility_target,
    ):
        self.features = features.float()
        self.pair_costs = pair_costs.float()
        self.utility_target = utility_target.float()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.pair_costs[idx],
            self.utility_target[idx],
        )


# ============================================================
# Training loss
# ============================================================

def compute_training_loss(
    predictor,
    features,
    pair_costs,
    utility_target,
    args,
    eps=1e-8,
):
    pair_scores, info = predictor(
        features,
        return_info=True,
    )

    utility_logits = info[
        "utility_logits"
    ]

    interaction_scores = info[
        "interaction_scores"
    ]

    min_cost = pair_costs.min(
        dim=-1,
        keepdim=True,
    ).values

    max_cost = pair_costs.max(
        dim=-1,
        keepdim=True,
    ).values

    spread = (
        max_cost - min_cost
    ).clamp_min(eps)

    normalized_cost = (
        pair_costs
        -
        min_cost
    ) / spread

    pair_prob = torch.softmax(
        pair_scores,
        dim=-1,
    )

    expected_normalized_regret = (
        pair_prob
        *
        normalized_cost
    ).sum(dim=-1).mean()

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
            utility_logits,
            dim=-1,
        ),
        utility_target,
        reduction="batchmean",
    )

    interaction_l2 = (
        interaction_scores.pow(2).mean()
    )

    total = (
        args.regret_weight
        *
        expected_normalized_regret
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


# ============================================================
# Generic pair predictor training
# ============================================================

@torch.no_grad()
def evaluate_pair_predictor(
    predictor,
    loader,
    device,
):
    predictor.eval()

    exact_all = []
    regret_all = []

    for (
        features,
        pair_costs,
        utility_target,
    ) in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        pair_costs = pair_costs.to(
            device,
            non_blocking=True,
        )

        pair_scores, _ = predictor(
            features,
            return_info=True,
        )

        pred = pair_scores.argmax(
            dim=-1
        )

        oracle = pair_costs.argmin(
            dim=-1
        )

        selected_cost = pair_costs.gather(
            dim=-1,
            index=pred[:, None],
        ).squeeze(-1)

        best_cost = pair_costs.min(
            dim=-1
        ).values

        exact_all.append(
            (pred == oracle)
            .float()
            .cpu()
        )

        regret_all.append(
            (
                selected_cost
                -
                best_cost
            ).cpu()
        )

    exact = torch.cat(exact_all)
    regret = torch.cat(regret_all)

    return {
        "exact":
            exact.mean().item(),

        "mean_regret":
            regret.mean().item(),

        "median_regret":
            regret.median().item(),
    }


def train_pair_predictor(
    predictor,
    train_dataset,
    val_dataset,
    device,
    args,
    epochs,
    name,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    before = evaluate_pair_predictor(
        predictor,
        val_loader,
        device,
    )

    print(
        f"\n{name} BEFORE training:"
    )

    print(
        f"  exact={100.0 * before['exact']:.2f}% | "
        f"regret={before['mean_regret']:.8e}"
    )

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    best_state = deepcopy(
        predictor.state_dict()
    )

    best_regret = (
        before["mean_regret"]
    )

    best_epoch = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        predictor.train()

        loss_sum = 0.0
        n = 0

        for (
            features,
            pair_costs,
            utility_target,
        ) in train_loader:
            features = features.to(
                device,
                non_blocking=True,
            )

            pair_costs = pair_costs.to(
                device,
                non_blocking=True,
            )

            utility_target = utility_target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_training_loss(
                predictor=predictor,
                features=features,
                pair_costs=pair_costs,
                utility_target=utility_target,
                args=args,
            )

            loss.backward()
            optimizer.step()

            B = features.shape[0]

            loss_sum += (
                loss.item() * B
            )

            n += B

        scheduler.step()

        val = evaluate_pair_predictor(
            predictor,
            val_loader,
            device,
        )

        print(
            f"{name} "
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={loss_sum/n:.6f} | "
            f"exact={100.0 * val['exact']:.2f}% | "
            f"regret={val['mean_regret']:.8e}"
        )

        if (
            val["mean_regret"]
            <
            best_regret
            -
            1e-12
        ):
            best_regret = (
                val["mean_regret"]
            )

            best_epoch = epoch

            best_state = deepcopy(
                predictor.state_dict()
            )

    predictor.load_state_dict(
        best_state,
        strict=True,
    )

    predictor.eval()

    after = evaluate_pair_predictor(
        predictor,
        val_loader,
        device,
    )

    print(
        f"\n{name} BEST:"
    )

    print(
        f"  epoch={best_epoch} | "
        f"exact={100.0 * after['exact']:.2f}% | "
        f"regret={after['mean_regret']:.8e}"
    )

    return before, after


# ============================================================
# Block-1 training set = ALL six actual Block-0 states
# ============================================================

def build_block1_dataset(
    split_data,
    combo_table,
    mini_heads,
):
    features = split_data[
        "block1_features"
    ]

    route_losses = split_data[
        "route_losses"
    ]

    # [N,6,H,F] -> [N*6,H,F]
    flat_features = features.reshape(
        -1,
        features.shape[2],
        features.shape[3],
    )

    # [N,6,6] -> [N*6,6]
    flat_costs = route_losses.reshape(
        -1,
        route_losses.shape[-1],
    )

    _, utility_target = (
        costs_to_utility_target(
            pair_costs=flat_costs,
            combo_table=combo_table,
            mini_heads=mini_heads,
        )
    )

    return PairCostDataset(
        flat_features,
        flat_costs,
        utility_target,
    )


# ============================================================
# Block-0 continuation cost under learned Block-1 policy
# ============================================================

@torch.no_grad()
def compute_block0_policy_costs(
    block1_predictor,
    split_data,
    device,
    batch_size,
):
    """
    For every image and every possible Block-0 pair p0:

        state1 = actual state produced by p0
        p1_hat = Block1Predictor(state1)
        cost0[p0] = exact final CE L[p0, p1_hat]

    This avoids training Block 0 against an unrealistic perfect continuation.
    """

    features = split_data[
        "block1_features"
    ].float()

    route_losses = split_data[
        "route_losses"
    ].float()

    N, P0, H, Fdim = (
        features.shape
    )

    flat_features = features.reshape(
        N * P0,
        H,
        Fdim,
    )

    pred_indices = []

    for start in range(
        0,
        flat_features.shape[0],
        batch_size,
    ):
        batch = flat_features[
            start:
            start + batch_size
        ].to(device)

        pair_scores, _ = block1_predictor(
            batch,
            return_info=True,
        )

        pred_indices.append(
            pair_scores.argmax(
                dim=-1
            ).cpu()
        )

    p1_hat = torch.cat(
        pred_indices,
        dim=0,
    ).reshape(
        N,
        P0,
    )

    policy_cost = route_losses.gather(
        dim=2,
        index=p1_hat[
            :,
            :,
            None,
        ],
    ).squeeze(-1)

    return policy_cost, p1_hat


def build_block0_dataset(
    split_data,
    block1_predictor,
    combo_table,
    mini_heads,
    device,
    batch_size,
):
    policy_costs, p1_hat = (
        compute_block0_policy_costs(
            block1_predictor=block1_predictor,
            split_data=split_data,
            device=device,
            batch_size=batch_size,
        )
    )

    _, utility_target = (
        costs_to_utility_target(
            pair_costs=policy_costs,
            combo_table=combo_table,
            mini_heads=mini_heads,
        )
    )

    dataset = PairCostDataset(
        split_data[
            "block0_features"
        ],
        policy_costs,
        utility_target,
    )

    return dataset, policy_costs, p1_hat


# ============================================================
# Sequential exact-cache evaluation
# ============================================================

@torch.no_grad()
def sequential_cache_eval(
    predictor0,
    predictor1,
    split_data,
    device,
    batch_size,
):
    b0_features = split_data[
        "block0_features"
    ].float()

    b1_features = split_data[
        "block1_features"
    ].float()

    route_losses = split_data[
        "route_losses"
    ].float()

    N = b0_features.shape[0]

    p0_all = []

    for start in range(
        0,
        N,
        batch_size,
    ):
        batch = b0_features[
            start:
            start + batch_size
        ].to(device)

        scores, _ = predictor0(
            batch,
            return_info=True,
        )

        p0_all.append(
            scores.argmax(
                dim=-1
            ).cpu()
        )

    p0 = torch.cat(
        p0_all,
        dim=0,
    )

    row = torch.arange(N)

    chosen_b1_features = b1_features[
        row,
        p0,
    ]

    p1_all = []

    for start in range(
        0,
        N,
        batch_size,
    ):
        batch = chosen_b1_features[
            start:
            start + batch_size
        ].to(device)

        scores, _ = predictor1(
            batch,
            return_info=True,
        )

        p1_all.append(
            scores.argmax(
                dim=-1
            ).cpu()
        )

    p1 = torch.cat(
        p1_all,
        dim=0,
    )

    selected_loss = route_losses[
        row,
        p0,
        p1,
    ]

    flat_losses = route_losses.reshape(
        N,
        -1,
    )

    oracle_loss, oracle_flat = (
        flat_losses.min(
            dim=-1
        )
    )

    oracle_p0 = (
        oracle_flat
        //
        route_losses.shape[2]
    )

    oracle_p1 = (
        oracle_flat
        %
        route_losses.shape[2]
    )

    whole_exact = (
        (p0 == oracle_p0)
        &
        (p1 == oracle_p1)
    ).float().mean().item()

    regret = (
        selected_loss
        -
        oracle_loss
    )

    return {
        "mean_ce":
            selected_loss.mean().item(),

        "oracle_ce":
            oracle_loss.mean().item(),

        "mean_regret":
            regret.mean().item(),

        "median_regret":
            regret.median().item(),

        "whole_exact":
            whole_exact,

        "p0":
            p0,

        "p1":
            p1,
    }


def best_static_from_train(
    train_data,
):
    mean_route = (
        train_data[
            "route_losses"
        ].mean(
            dim=0
        )
    )

    flat_idx = int(
        mean_route.reshape(-1)
        .argmin()
        .item()
    )

    p1_count = mean_route.shape[1]

    p0 = flat_idx // p1_count
    p1 = flat_idx % p1_count

    return p0, p1


def static_eval(
    split_data,
    p0,
    p1,
):
    losses = split_data[
        "route_losses"
    ][
        :,
        p0,
        p1,
    ]

    return losses.mean().item()


def oracle_eval(
    split_data,
):
    flat = split_data[
        "route_losses"
    ].reshape(
        split_data[
            "route_losses"
        ].shape[0],
        -1,
    )

    return (
        flat.min(
            dim=-1
        ).values.mean().item()
    )


def print_system_eval(
    title,
    metrics,
    static_ce,
):
    oracle_gap = (
        static_ce
        -
        metrics[
            "oracle_ce"
        ]
    )

    dynamic_gain = (
        static_ce
        -
        metrics[
            "mean_ce"
        ]
    )

    if oracle_gap > 0:
        gap_capture = (
            100.0
            *
            dynamic_gain
            /
            oracle_gap
        )
    else:
        gap_capture = float("nan")

    print(
        f"\n{title}"
    )

    print(
        f"  sequential CE: "
        f"{metrics['mean_ce']:.6f}"
    )

    print(
        f"  oracle CE:     "
        f"{metrics['oracle_ce']:.6f}"
    )

    print(
        f"  mean regret:   "
        f"{metrics['mean_regret']:.8e}"
    )

    print(
        f"  whole-route exact: "
        f"{100.0 * metrics['whole_exact']:.2f}%"
    )

    print(
        f"  captured Static→Oracle CE gap: "
        f"{gap_capture:.2f}%"
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
        "AMP teacher generation:",
        args.amp,
    )

    print(
        "\nStage 3 goal:"
    )

    print(
        "Train routing from FINAL classification loss of all 36 whole-network routes."
    )

    print(
        "Block 1 learns final pair value from every possible Block-0 state."
    )

    print(
        "Block 0 then learns continuation value under the learned Block-1 policy."
    )

    backbone = load_file(
        args.backbone_checkpoint,
        device,
        "Seed-scale tuned backbone checkpoint",
    )

    model = build_model_from_config(
        backbone[
            "config"
        ]
    ).to(device)

    model.load_state_dict(
        backbone[
            "model"
        ],
        strict=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if model.depth != 2:
        raise ValueError(
            f"Current Stage-3 exact routing supports depth=2; got {model.depth}."
        )

    combinations = list(
        itertools.combinations(
            range(model.mini_heads),
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

    print(
        "Whole-network routes:",
        len(combinations) ** model.depth,
    )

    init_checkpoint = load_file(
        args.init_predictor_checkpoint,
        device,
        "Initial Utility+Interaction predictor",
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

    predictors, hidden_dim, dropout = (
        load_predictors(
            checkpoint=init_checkpoint,
            model=model,
            feature_dim=feature_dim,
            device=device,
        )
    )

    predictor0 = predictors[0]
    predictor1 = predictors[1]

    train_set, val_set = (
        build_global_value_sets(
            args
        )
    )

    train_data, val_data = (
        load_or_build_cache(
            model=model,
            train_set=train_set,
            val_set=val_set,
            combinations=combinations,
            device=device,
            args=args,
        )
    )

    print(
        "\nCache shapes:"
    )

    print(
        "train block0 features:",
        tuple(
            train_data[
                "block0_features"
            ].shape
        ),
    )

    print(
        "train block1 features:",
        tuple(
            train_data[
                "block1_features"
            ].shape
        ),
    )

    print(
        "train route losses:",
        tuple(
            train_data[
                "route_losses"
            ].shape
        ),
    )

    print(
        "val route losses:",
        tuple(
            val_data[
                "route_losses"
            ].shape
        ),
    )

    # --------------------------------------------------------
    # Fixed route baseline selected only on Stage-3 train.
    # --------------------------------------------------------

    static_p0, static_p1 = (
        best_static_from_train(
            train_data
        )
    )

    static_val_ce = static_eval(
        val_data,
        static_p0,
        static_p1,
    )

    oracle_val_ce = oracle_eval(
        val_data
    )

    print(
        "\n================ STAGE-3 VAL BASELINES ================"
    )

    print(
        f"Best fixed route from train: "
        f"Block0={combinations[static_p0]}, "
        f"Block1={combinations[static_p1]}"
    )

    print(
        f"Static val CE: "
        f"{static_val_ce:.6f}"
    )

    print(
        f"Oracle val CE: "
        f"{oracle_val_ce:.6f}"
    )

    print(
        f"Available Static→Oracle gap: "
        f"{static_val_ce - oracle_val_ce:.8f}"
    )

    # --------------------------------------------------------
    # Existing predictor on the exact same fresh val cache.
    # --------------------------------------------------------

    old_val = sequential_cache_eval(
        predictor0=predictor0,
        predictor1=predictor1,
        split_data=val_data,
        device=device,
        batch_size=args.train_batch_size,
    )

    print_system_eval(
        "CURRENT PREDICTOR on fresh Stage-3 val",
        old_val,
        static_val_ce,
    )

    # --------------------------------------------------------
    # STEP 1: Train Block 1 from final task loss for ALL p0 states.
    # --------------------------------------------------------

    train_b1_dataset = build_block1_dataset(
        split_data=train_data,
        combo_table=combo_table,
        mini_heads=model.mini_heads,
    )

    val_b1_dataset = build_block1_dataset(
        split_data=val_data,
        combo_table=combo_table,
        mini_heads=model.mini_heads,
    )

    print(
        "\n================ TRAIN BLOCK 1 GLOBAL VALUE ================"
    )

    print(
        f"Block-1 train states: "
        f"{len(train_b1_dataset)} "
        f"(images × 6 possible Block-0 states)"
    )

    train_pair_predictor(
        predictor=predictor1,
        train_dataset=train_b1_dataset,
        val_dataset=val_b1_dataset,
        device=device,
        args=args,
        epochs=args.block1_epochs,
        name="Block1",
    )

    # --------------------------------------------------------
    # STEP 2: Back up learned Block-1 continuation to Block 0.
    # --------------------------------------------------------

    train_b0_dataset, train_policy_costs, _ = (
        build_block0_dataset(
            split_data=train_data,
            block1_predictor=predictor1,
            combo_table=combo_table,
            mini_heads=model.mini_heads,
            device=device,
            batch_size=args.train_batch_size,
        )
    )

    val_b0_dataset, val_policy_costs, _ = (
        build_block0_dataset(
            split_data=val_data,
            block1_predictor=predictor1,
            combo_table=combo_table,
            mini_heads=model.mini_heads,
            device=device,
            batch_size=args.train_batch_size,
        )
    )

    print(
        "\n================ TRAIN BLOCK 0 CONTINUATION VALUE ================"
    )

    print(
        "Block-0 target = final CE after the trained Block-1 policy continues."
    )

    train_pair_predictor(
        predictor=predictor0,
        train_dataset=train_b0_dataset,
        val_dataset=val_b0_dataset,
        device=device,
        args=args,
        epochs=args.block0_epochs,
        name="Block0",
    )

    # --------------------------------------------------------
    # Sequential policy evaluation on fresh Stage-3 val.
    # --------------------------------------------------------

    new_val = sequential_cache_eval(
        predictor0=predictor0,
        predictor1=predictor1,
        split_data=val_data,
        device=device,
        batch_size=args.train_batch_size,
    )

    print(
        "\n================ GLOBAL-VALUE RESULT ================"
    )

    print_system_eval(
        "OLD predictor",
        old_val,
        static_val_ce,
    )

    print_system_eval(
        "NEW global-value predictor",
        new_val,
        static_val_ce,
    )

    print(
        "\nDirect comparison:"
    )

    print(
        f"  CE: "
        f"{old_val['mean_ce']:.6f}"
        " -> "
        f"{new_val['mean_ce']:.6f}"
    )

    print(
        f"  Mean regret: "
        f"{old_val['mean_regret']:.8e}"
        " -> "
        f"{new_val['mean_regret']:.8e}"
    )

    print(
        f"  Whole-route exact: "
        f"{100.0 * old_val['whole_exact']:.2f}%"
        " -> "
        f"{100.0 * new_val['whole_exact']:.2f}%"
    )

    # --------------------------------------------------------
    # Save both predictors in the same format used by eval scripts.
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output_checkpoint)
        or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "predictors": {
                "block_0":
                    predictor0.state_dict(),

                "block_1":
                    predictor1.state_dict(),
            },

            "config": {
                "hidden_dim":
                    hidden_dim,

                "dropout":
                    dropout,

                "training":
                    "global_final_value_backward_induction",
            },

            "source_backbone_checkpoint":
                args.backbone_checkpoint,

            "source_init_predictor_checkpoint":
                args.init_predictor_checkpoint,

            "route_cache":
                args.route_cache,

            "val_metrics": {
                "static_ce":
                    static_val_ce,

                "oracle_ce":
                    oracle_val_ce,

                "old_dynamic_ce":
                    old_val[
                        "mean_ce"
                    ],

                "new_dynamic_ce":
                    new_val[
                        "mean_ce"
                    ],

                "old_regret":
                    old_val[
                        "mean_regret"
                    ],

                "new_regret":
                    new_val[
                        "mean_regret"
                    ],
            },
        },
        args.output_checkpoint,
    )

    print(
        "\nSaved global-value predictor:"
    )

    print(
        args.output_checkpoint
    )

    # Explicit preliminary decision.
    old_gain = (
        static_val_ce
        -
        old_val[
            "mean_ce"
        ]
    )

    new_gain = (
        static_val_ce
        -
        new_val[
            "mean_ce"
        ]
    )

    available = (
        static_val_ce
        -
        oracle_val_ce
    )

    print(
        "\n================ PRELIMINARY STAGE-3 VERDICT ================"
    )

    print(
        f"Old dynamic gain over Static: "
        f"{old_gain:.8f}"
    )

    print(
        f"New dynamic gain over Static: "
        f"{new_gain:.8f}"
    )

    if available > 0:
        print(
            f"Old captured oracle gap: "
            f"{100.0 * old_gain / available:.2f}%"
        )

        print(
            f"New captured oracle gap: "
            f"{100.0 * new_gain / available:.2f}%"
        )

    if (
        new_val[
            "mean_ce"
        ]
        <
        old_val[
            "mean_ce"
        ]
    ):
        print(
            "PASS: final-task global-value teacher improved sequential route selection "
            "on Stage-3 validation."
        )

        print(
            "Next step: run ONE completely fresh held-out Dynamic-vs-Static-vs-Oracle test."
        )

    else:
        print(
            "FAIL: global-value teacher did not improve sequential route selection "
            "on Stage-3 validation."
        )

        print(
            "Do not spend a fresh held-out split yet; analyze predictor inputs/capacity first."
        )


if __name__ == "__main__":
    main()
