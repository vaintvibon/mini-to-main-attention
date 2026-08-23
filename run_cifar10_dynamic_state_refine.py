import argparse
import itertools
import math
import os
import random
from collections import Counter
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.utility_interaction_predictor import UtilityInteractionPredictor


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

    p.add_argument(
        "--output-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument(
        "--state-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_block1_cache.pt"
        ),
    )

    p.add_argument(
        "--rebuild-state-cache",
        action="store_true",
    )

    p.add_argument("--seed", type=int, default=42)

    # Same splits as prior Stage-2.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)

    p.add_argument("--teacher-batch-size", type=int, default=64)
    p.add_argument("--train-batch-size", type=int, default=128)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--regret-weight", type=float, default=1.0)
    p.add_argument("--pair-kl-weight", type=float, default=0.5)
    p.add_argument("--utility-kl-weight", type=float, default=0.5)
    p.add_argument("--interaction-l2-weight", type=float, default=0.01)
    p.add_argument("--pair-temperature", type=float, default=0.5)

    p.add_argument("--num-workers", type=int, default=2)
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
# Dataset
# ============================================================

def eval_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_stage2_sets(args):
    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=eval_transform(),
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
    )

    train_indices = permutation[
        start:
        start + args.utility_train_subset
    ]

    start += args.utility_train_subset

    val_indices = permutation[
        start:
        start + args.utility_val_subset
    ]

    return (
        Subset(base, train_indices),
        Subset(base, val_indices),
    )


# ============================================================
# Mini feature
# ============================================================

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
def choose_block0_pair(
    model,
    predictor0,
    x0,
    combo_table,
):
    block0 = model.blocks[0]

    x_norm = block0.norm1(x0)

    mini_contexts, mini_attn = (
        block0.attn.mini_attention(
            x_norm,
            patch_hw=model.patch_hw,
        )
    )

    features = extract_mini_features(
        mini_contexts,
        mini_attn,
    )

    pair_scores, _ = predictor0(
        features,
        return_info=True,
    )

    combo_idx = pair_scores.argmax(dim=-1)

    return (
        combo_table.to(x0.device)[combo_idx],
        combo_idx,
    )


@torch.no_grad()
def execute_block0(
    model,
    x0,
    block0_pair,
):
    return model.blocks[0](
        x0,
        patch_hw=model.patch_hw,
        return_info=False,
        collect_taylor=False,
        forced_direct_indices=block0_pair,
        forced_uniform_mix=True,
    )


@torch.no_grad()
def extract_block1_state_features(
    model,
    x1,
):
    block1 = model.blocks[1]

    x_norm = block1.norm1(x1)

    mini_contexts, mini_attn = (
        block1.attn.mini_attention(
            x_norm,
            patch_hw=model.patch_hw,
        )
    )

    return extract_mini_features(
        mini_contexts,
        mini_attn,
    )


@torch.no_grad()
def evaluate_block1_candidates(
    model,
    x1,
    labels,
    combo_table,
):
    """
    Current Block-0 route is already reflected in x1.

    Evaluate all six Block-1 Direct pairs from THIS ACTUAL STATE.
    """

    block1 = model.blocks[1]
    combo_device = combo_table.to(x1.device)

    B = x1.shape[0]

    losses = []

    for combo_idx in range(
        combo_device.shape[0]
    ):
        pair = (
            combo_device[
                combo_idx
            ][None, :]
            .expand(B, -1)
            .clone()
        )

        x2 = block1(
            x1,
            patch_hw=model.patch_hw,
            return_info=False,
            collect_taylor=False,
            forced_direct_indices=pair,
            forced_uniform_mix=True,
        )

        logits = model.head(
            model.norm(x2)[:, 0]
        )

        per_sample_loss = F.cross_entropy(
            logits.float(),
            labels,
            reduction="none",
        )

        losses.append(
            per_sample_loss
        )

    return torch.stack(
        losses,
        dim=-1,
    )


# ============================================================
# Utility teacher from actual-state pair losses
# ============================================================

def pair_losses_to_utility_target(
    subset_losses,
    combo_table,
    mini_heads,
    eps=1e-8,
):
    """
    U_h = mean(loss | h excluded) - mean(loss | h included)

    Then center/std-normalize across heads and softmax.
    """

    utilities = []

    combo_table_cpu = combo_table.cpu()

    for head_idx in range(mini_heads):
        included = (
            combo_table_cpu
            ==
            head_idx
        ).any(dim=-1)

        excluded = ~included

        include_loss = (
            subset_losses[
                :,
                included,
            ].mean(dim=-1)
        )

        exclude_loss = (
            subset_losses[
                :,
                excluded,
            ].mean(dim=-1)
        )

        utilities.append(
            exclude_loss
            -
            include_loss
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

    # If all utilities are exactly equal, use a uniform target.
    zero_signal = (
        std.squeeze(-1)
        <=
        eps
    )

    target = torch.softmax(
        normalized,
        dim=-1,
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
# Actual-state cache
# ============================================================

@torch.no_grad()
def generate_actual_state_split(
    model,
    predictor0,
    dataset,
    combo_table,
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
            device.type == "cuda"
        ),
        drop_last=False,
    )

    feature_all = []
    loss_all = []
    utility_all = []
    utility_target_all = []
    block0_idx_all = []

    seen = 0
    total = len(dataset)

    print(
        f"\nGenerating actual-state Block-1 teacher: "
        f"{split_name}"
    )

    model.eval()
    predictor0.eval()

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
            x0 = prepare_tokens(
                model,
                images,
            )

            block0_pair, block0_idx = (
                choose_block0_pair(
                    model=model,
                    predictor0=predictor0,
                    x0=x0,
                    combo_table=combo_table,
                )
            )

            x1 = execute_block0(
                model,
                x0,
                block0_pair,
            )

            block1_features = (
                extract_block1_state_features(
                    model,
                    x1,
                )
            )

            subset_losses = (
                evaluate_block1_candidates(
                    model=model,
                    x1=x1,
                    labels=labels,
                    combo_table=combo_table,
                )
            )

        subset_losses_cpu = (
            subset_losses.float().cpu()
        )

        utility, utility_target = (
            pair_losses_to_utility_target(
                subset_losses=subset_losses_cpu,
                combo_table=combo_table,
                mini_heads=model.mini_heads,
            )
        )

        feature_all.append(
            block1_features.float().cpu()
        )

        loss_all.append(
            subset_losses_cpu
        )

        utility_all.append(
            utility
        )

        utility_target_all.append(
            utility_target
        )

        block0_idx_all.append(
            block0_idx.cpu()
        )

        seen += images.shape[0]

        print(
            f"{split_name}: "
            f"{seen}/{total}"
        )

    return {
        "features":
            torch.cat(
                feature_all,
                dim=0,
            ),

        "subset_losses":
            torch.cat(
                loss_all,
                dim=0,
            ),

        "head_utility":
            torch.cat(
                utility_all,
                dim=0,
            ),

        "teacher_target":
            torch.cat(
                utility_target_all,
                dim=0,
            ),

        "block0_combo_idx":
            torch.cat(
                block0_idx_all,
                dim=0,
            ),
    }


def load_or_build_state_cache(
    model,
    predictor0,
    train_set,
    val_set,
    combo_table,
    device,
    args,
):
    if (
        os.path.exists(args.state_cache)
        and
        not args.rebuild_state_cache
    ):
        print(
            "\nLoading actual-state cache:"
        )
        print(
            args.state_cache
        )

        cache = torch.load(
            args.state_cache,
            map_location="cpu",
            weights_only=False,
        )

        return cache["train"], cache["val"]

    train_data = generate_actual_state_split(
        model=model,
        predictor0=predictor0,
        dataset=train_set,
        combo_table=combo_table,
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

    val_data = generate_actual_state_split(
        model=model,
        predictor0=predictor0,
        dataset=val_set,
        combo_table=combo_table,
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
        os.path.dirname(args.state_cache) or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "train": train_data,
            "val": val_data,
            "source_predictor_checkpoint":
                args.predictor_checkpoint,
        },
        args.state_cache,
    )

    print(
        "\nSaved actual-state cache:"
    )
    print(
        args.state_cache
    )

    return train_data, val_data


# ============================================================
# Offline dataset
# ============================================================

class Block1Dataset(Dataset):
    def __init__(
        self,
        features,
        subset_losses,
        teacher_target,
    ):
        self.features = features.float()
        self.subset_losses = subset_losses.float()
        self.teacher_target = teacher_target.float()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.subset_losses[idx],
            self.teacher_target[idx],
        )


# ============================================================
# Loss / metrics
# ============================================================

def compute_loss(
    predictor,
    features,
    subset_losses,
    teacher_target,
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

    interactions = info[
        "interaction_scores"
    ]

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
    ).clamp_min(eps)

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
        teacher_target,
        reduction="batchmean",
    )

    interaction_l2 = (
        interactions.pow(2).mean()
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
        "reg":
            expected_regret.detach(),

        "pair_kl":
            pair_kl.detach(),

        "utility_kl":
            utility_kl.detach(),

        "interaction_l2":
            interaction_l2.detach(),
    }


def head_overlap(
    pair_a,
    pair_b,
):
    match = (
        pair_a[..., :, None]
        ==
        pair_b[..., None, :]
    )

    return (
        match.any(dim=-1)
        .float()
        .mean(dim=-1)
    )


@torch.no_grad()
def evaluate_offline(
    predictor,
    loader,
    combo_table,
    device,
):
    predictor.eval()

    combo_device = combo_table.to(
        device
    )

    exact_all = []
    overlap_all = []
    regret_all = []
    selected_loss_all = []
    pred_idx_all = []

    utility_top1_all = []
    utility_top2_teacher_all = []

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

        pair_scores, info = predictor(
            features,
            return_info=True,
        )

        pred_idx = pair_scores.argmax(
            dim=-1
        )

        oracle_idx = subset_losses.argmin(
            dim=-1
        )

        exact = (
            pred_idx == oracle_idx
        ).float()

        pred_pair = combo_device[
            pred_idx
        ]

        oracle_pair = combo_device[
            oracle_idx
        ]

        overlap = head_overlap(
            pred_pair,
            oracle_pair,
        )

        selected_loss = (
            subset_losses.gather(
                dim=-1,
                index=pred_idx[:, None],
            ).squeeze(-1)
        )

        oracle_loss = subset_losses.min(
            dim=-1
        ).values

        regret = (
            selected_loss
            -
            oracle_loss
        )

        pred_utility = info[
            "utility_logits"
        ]

        pred_top1 = pred_utility.argmax(
            dim=-1
        )

        teacher_top1 = teacher_target.argmax(
            dim=-1
        )

        utility_top1 = (
            pred_top1 == teacher_top1
        ).float()

        pred_top2 = torch.topk(
            pred_utility,
            k=2,
            dim=-1,
        ).indices.sort(dim=-1).values

        teacher_top2 = torch.topk(
            teacher_target,
            k=2,
            dim=-1,
        ).indices.sort(dim=-1).values

        utility_top2_teacher = (
            pred_top2 == teacher_top2
        ).all(dim=-1).float()

        exact_all.append(
            exact.cpu()
        )

        overlap_all.append(
            overlap.cpu()
        )

        regret_all.append(
            regret.cpu()
        )

        selected_loss_all.append(
            selected_loss.cpu()
        )

        pred_idx_all.append(
            pred_idx.cpu()
        )

        utility_top1_all.append(
            utility_top1.cpu()
        )

        utility_top2_teacher_all.append(
            utility_top2_teacher.cpu()
        )

    exact = torch.cat(
        exact_all
    )

    overlap = torch.cat(
        overlap_all
    )

    regret = torch.cat(
        regret_all
    )

    selected_loss = torch.cat(
        selected_loss_all
    )

    return {
        "exact":
            exact.mean().item(),

        "overlap":
            overlap.mean().item(),

        "mean_regret":
            regret.mean().item(),

        "median_regret":
            regret.median().item(),

        "mean_selected_ce":
            selected_loss.mean().item(),

        "utility_top1":
            torch.cat(
                utility_top1_all
            ).mean().item(),

        "utility_top2_teacher":
            torch.cat(
                utility_top2_teacher_all
            ).mean().item(),

        "pred_idx":
            torch.cat(
                pred_idx_all
            ),
    }


def print_metrics(
    title,
    metrics,
):
    print(
        f"\n{title}"
    )

    print(
        f"Pair exact: "
        f"{100.0 * metrics['exact']:.2f}%"
    )

    print(
        f"Head overlap: "
        f"{100.0 * metrics['overlap']:.2f}%"
    )

    print(
        f"Mean regret: "
        f"{metrics['mean_regret']:.8e}"
    )

    print(
        f"Mean selected CE: "
        f"{metrics['mean_selected_ce']:.6f}"
    )

    print(
        f"Utility teacher Top-1: "
        f"{100.0 * metrics['utility_top1']:.2f}%"
    )

    print(
        f"Utility teacher Top-2 exact: "
        f"{100.0 * metrics['utility_top2_teacher']:.2f}%"
    )


# ============================================================
# Train ONLY Block 1
# ============================================================

def train_block1(
    predictors,
    train_loader,
    val_loader,
    combo_table,
    device,
    args,
    hidden_dim,
    dropout,
):
    predictor0 = predictors[0]
    predictor1 = predictors[1]

    # Freeze Block 0 predictor completely.
    for p in predictor0.parameters():
        p.requires_grad_(False)

    predictor0.eval()

    for p in predictor1.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        predictor1.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    before = evaluate_offline(
        predictor1,
        val_loader,
        combo_table,
        device,
    )

    print_metrics(
        "Block-1 BEFORE actual-state refinement",
        before,
    )

    best_regret = math.inf
    best_exact = -1.0
    best_epoch = -1

    os.makedirs(
        os.path.dirname(
            args.output_checkpoint
        )
        or ".",
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        predictor1.train()

        sums = {
            "loss": 0.0,
            "reg": 0.0,
            "pair_kl": 0.0,
            "utility_kl": 0.0,
            "int_l2": 0.0,
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

            loss, li = compute_loss(
                predictor=predictor1,
                features=features,
                subset_losses=subset_losses,
                teacher_target=teacher_target,
                args=args,
            )

            loss.backward()
            optimizer.step()

            B = features.shape[0]

            sums["loss"] += (
                loss.item() * B
            )
            sums["reg"] += (
                li["reg"].item() * B
            )
            sums["pair_kl"] += (
                li["pair_kl"].item() * B
            )
            sums["utility_kl"] += (
                li["utility_kl"].item() * B
            )
            sums["int_l2"] += (
                li["interaction_l2"].item() * B
            )
            sums["n"] += B

        scheduler.step()

        val = evaluate_offline(
            predictor1,
            val_loader,
            combo_table,
            device,
        )

        n = sums["n"]

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"loss={sums['loss']/n:.6f} | "
            f"reg={sums['reg']/n:.6f} | "
            f"pairKL={sums['pair_kl']/n:.6f} | "
            f"utilityKL={sums['utility_kl']/n:.6f} | "
            f"intL2={sums['int_l2']/n:.6f}"
        )

        print(
            f"Val exact="
            f"{100.0 * val['exact']:.2f}% | "
            f"overlap="
            f"{100.0 * val['overlap']:.2f}% | "
            f"regret="
            f"{val['mean_regret']:.8e} | "
            f"CE="
            f"{val['mean_selected_ce']:.6f}"
        )

        better = (
            val["mean_regret"]
            <
            best_regret
            -
            1e-12
        )

        if (
            not better
            and
            abs(
                val["mean_regret"]
                -
                best_regret
            )
            <=
            1e-12
            and
            val["exact"]
            >
            best_exact
        ):
            better = True

        if better:
            best_regret = (
                val["mean_regret"]
            )

            best_exact = (
                val["exact"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "predictors": {
                        "block_0":
                            predictor0.state_dict(),

                        "block_1":
                            predictor1.state_dict(),
                    },

                    "best_epoch":
                        best_epoch,

                    "best_val_regret":
                        best_regret,

                    "best_val_exact":
                        best_exact,

                    "config": {
                        "hidden_dim":
                            hidden_dim,

                        "dropout":
                            dropout,

                        "training":
                            "block1_actual_dynamic_state_refinement",
                    },

                    "source_predictor_checkpoint":
                        args.predictor_checkpoint,
                },
                args.output_checkpoint,
            )

    best = torch.load(
        args.output_checkpoint,
        map_location=device,
        weights_only=False,
    )

    predictor1.load_state_dict(
        best["predictors"]["block_1"],
        strict=True,
    )

    predictor1.eval()

    after = evaluate_offline(
        predictor1,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\nBest checkpoint:"
    )
    print(
        args.output_checkpoint
    )
    print(
        f"Best epoch: {best_epoch}"
    )

    print_metrics(
        "Block-1 AFTER actual-state refinement",
        after,
    )

    return before, after


# ============================================================
# Frequency
# ============================================================

def print_pair_frequency(
    title,
    pred_idx,
    combinations,
):
    print(
        f"\n{title}"
    )

    counter = Counter(
        int(v)
        for v in pred_idx.tolist()
    )

    total = pred_idx.shape[0]

    for idx, combo in enumerate(
        combinations
    ):
        count = counter.get(
            idx,
            0,
        )

        print(
            f"  {combo}: "
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

    stage1 = load_file(
        args.stage1_checkpoint,
        device,
        "Stage-1 checkpoint",
    )

    model = build_model_from_stage1_config(
        stage1["config"]
    ).to(device)

    model.load_state_dict(
        stage1["model"],
        strict=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if model.depth != 2:
        raise ValueError(
            "This refinement experiment intentionally supports depth=2 only. "
            f"Current depth={model.depth}"
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

    predictor_ckpt = load_file(
        args.predictor_checkpoint,
        device,
        "Utility+Interaction predictor checkpoint",
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
            checkpoint=predictor_ckpt,
            model=model,
            feature_dim=feature_dim,
            device=device,
        )
    )

    print(
        "\nLoaded Stage-1 backbone:"
    )
    print(
        args.stage1_checkpoint
    )

    print(
        "\nLoaded current Utility + Interaction predictors:"
    )
    print(
        args.predictor_checkpoint
    )

    print(
        "\nExperiment rule:"
    )
    print(
        "Block 0 predictor is frozen."
    )
    print(
        "Block 1 is retrained from the ACTUAL state produced by Block 0's dynamic choice."
    )

    train_set, val_set = (
        build_stage2_sets(
            args
        )
    )

    train_data, val_data = (
        load_or_build_state_cache(
            model=model,
            predictor0=predictors[0],
            train_set=train_set,
            val_set=val_set,
            combo_table=combo_table,
            device=device,
            args=args,
        )
    )

    print(
        "\nActual-state cache shapes:"
    )
    print(
        "train features:",
        tuple(
            train_data["features"].shape
        ),
    )
    print(
        "train subset losses:",
        tuple(
            train_data["subset_losses"].shape
        ),
    )
    print(
        "val features:",
        tuple(
            val_data["features"].shape
        ),
    )

    print(
        "\nBlock-0 dynamic pair distribution used to generate Block-1 states:"
    )

    print_pair_frequency(
        "Train Block-0 choices",
        train_data[
            "block0_combo_idx"
        ],
        combinations,
    )

    print_pair_frequency(
        "Val Block-0 choices",
        val_data[
            "block0_combo_idx"
        ],
        combinations,
    )

    train_ds = Block1Dataset(
        train_data["features"],
        train_data["subset_losses"],
        train_data["teacher_target"],
    )

    val_ds = Block1Dataset(
        val_data["features"],
        val_data["subset_losses"],
        val_data["teacher_target"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    before, after = train_block1(
        predictors=predictors,
        train_loader=train_loader,
        val_loader=val_loader,
        combo_table=combo_table,
        device=device,
        args=args,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    print_pair_frequency(
        "Block-1 choices BEFORE refinement",
        before["pred_idx"],
        combinations,
    )

    print_pair_frequency(
        "Block-1 choices AFTER refinement",
        after["pred_idx"],
        combinations,
    )

    print(
        "\n================ SUMMARY ================"
    )

    print(
        "Block 0 was unchanged."
    )

    print(
        "Only Block 1 was adapted to states actually produced by dynamic Block-0 routing."
    )

    print(
        "\nExact:"
    )
    print(
        f"{100.0 * before['exact']:.2f}%"
        " -> "
        f"{100.0 * after['exact']:.2f}%"
    )

    print(
        "\nOverlap:"
    )
    print(
        f"{100.0 * before['overlap']:.2f}%"
        " -> "
        f"{100.0 * after['overlap']:.2f}%"
    )

    print(
        "\nMean regret:"
    )
    print(
        f"{before['mean_regret']:.8e}"
        " -> "
        f"{after['mean_regret']:.8e}"
    )

    print(
        "\nMean selected CE:"
    )
    print(
        f"{before['mean_selected_ce']:.6f}"
        " -> "
        f"{after['mean_selected_ce']:.6f}"
    )

    print(
        "\nNext:"
    )

    print(
        "Use the saved refined checkpoint in run_cifar10_final_routing_eval.py "
        "to re-run the official-test comparison."
    )


if __name__ == "__main__":
    main()
