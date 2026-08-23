import argparse
import math
import os
import random
from copy import deepcopy

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

    p.add_argument(
        "--output-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage1_cifar10_seedscale_tuned.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # Previously consumed train-split regions.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)
    p.add_argument("--diagnostic-subset", type=int, default=1000)

    # Fresh data for this experiment.
    p.add_argument("--scale-train-subset", type=int, default=1000)
    p.add_argument("--scale-val-subset", type=int, default=500)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # Small regularizer around the original learned scale.
    # Set 0 to remove it completely.
    p.add_argument("--scale-reg-weight", type=float, default=1e-4)

    # Keep the experiment interpretable: positive finite seed strength.
    p.add_argument("--min-scale", type=float, default=0.0)
    p.add_argument("--max-scale", type=float, default=8.0)

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Loading
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

        if key not in states:
            raise KeyError(
                f"Missing predictor state: {key}"
            )

        predictor.load_state_dict(
            states[key],
            strict=True,
        )

        predictor.eval()

        for parameter in predictor.parameters():
            parameter.requires_grad_(False)

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
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )


def build_scale_datasets(args):
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
    )

    train_end = (
        start
        +
        args.scale_train_subset
    )

    val_end = (
        train_end
        +
        args.scale_val_subset
    )

    if val_end > len(base):
        raise ValueError(
            f"Requested split ends at {val_end}, "
            f"but CIFAR-10 train has only {len(base)} samples."
        )

    train_indices = permutation[
        start:train_end
    ]

    val_indices = permutation[
        train_end:val_end
    ]

    print(
        "\nFresh seed-scale split:"
    )

    print(
        f"  permutation offset train: "
        f"[{start}, {train_end})"
    )

    print(
        f"  permutation offset val:   "
        f"[{train_end}, {val_end})"
    )

    print(
        "  official CIFAR-10 test is NOT used."
    )

    return (
        Subset(
            base,
            train_indices,
        ),
        Subset(
            base,
            val_indices,
        ),
    )


# ============================================================
# Mini feature / routing
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


def build_combo_table(
    model,
    device,
):
    import itertools

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
        device=device,
    )

    return combinations, combo_table


def choose_pair_no_grad(
    model,
    block,
    predictor,
    x,
    combo_table,
):
    """
    Hard routing is intentionally treated as a fixed decision.
    Gradients are NOT sent through argmax/selector.

    The actual Transformer block is then executed with gradients,
    so CE can update only seed_scale.
    """

    with torch.no_grad():
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

        pair_scores, _ = predictor(
            features,
            return_info=True,
        )

        combo_idx = pair_scores.argmax(
            dim=-1
        )

        forced_pair = combo_table[
            combo_idx
        ]

    return forced_pair


def forward_dynamic_for_scale_training(
    model,
    predictors,
    images,
    combo_table,
):
    """
    Sequential actual dynamic routing:

        block input
        -> frozen predictor picks pair
        -> execute block with gradients only to seed_scale
        -> changed state enters next block
        -> next frozen predictor picks from that actual state
    """

    x = prepare_tokens(
        model,
        images,
    )

    for block_idx, block in enumerate(
        model.blocks
    ):
        forced_pair = choose_pair_no_grad(
            model=model,
            block=block,
            predictor=predictors[
                block_idx
            ],
            x=x,
            combo_table=combo_table,
        )

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

    logits = model.head(
        x[:, 0]
    )

    return logits


# ============================================================
# Only six seed_scale parameters are trainable
# ============================================================

def get_seed_scale_parameters(
    model,
):
    params = []

    names = []

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
                f"Block {block_idx} main attention has no seed_scale."
            )

        parameter = (
            main_attention.seed_scale
        )

        params.append(
            parameter
        )

        names.append(
            f"blocks.{block_idx}."
            f"attn.main_attention.seed_scale"
        )

    return names, params


def freeze_everything_except_seed_scale(
    model,
):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    names, params = (
        get_seed_scale_parameters(
            model
        )
    )

    for parameter in params:
        parameter.requires_grad_(True)

    trainable = [
        (
            name,
            parameter.numel(),
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    total_trainable = sum(
        count
        for _, count in trainable
    )

    print(
        "\nTrainable parameters:"
    )

    for name, count in trainable:
        print(
            f"  {name}: {count}"
        )

    print(
        f"Total trainable parameters: "
        f"{total_trainable}"
    )

    return names, params


def print_scales(
    title,
    params,
):
    print(
        f"\n{title}"
    )

    for block_idx, parameter in enumerate(
        params
    ):
        values = (
            parameter
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        print(
            f"  Block {block_idx}: "
            f"{values}"
        )


def clamp_scales(
    params,
    min_scale,
    max_scale,
):
    with torch.no_grad():
        for parameter in params:
            parameter.clamp_(
                min=min_scale,
                max=max_scale,
            )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    predictors,
    loader,
    combo_table,
    device,
):
    model.eval()

    for predictor in predictors:
        predictor.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = forward_dynamic_for_scale_training(
            model=model,
            predictors=predictors,
            images=images,
            combo_table=combo_table,
        )

        loss = F.cross_entropy(
            logits.float(),
            labels,
            reduction="sum",
        )

        total_loss += loss.item()

        total_correct += (
            logits.argmax(
                dim=-1
            )
            ==
            labels
        ).sum().item()

        total_samples += labels.numel()

    return {
        "ce":
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


# ============================================================
# Train
# ============================================================

def train_seed_scale(
    model,
    predictors,
    train_loader,
    val_loader,
    combo_table,
    device,
    args,
    stage1_checkpoint,
):
    names, params = (
        freeze_everything_except_seed_scale(
            model
        )
    )

    original_scales = [
        p.detach().clone()
        for p in params
    ]

    print_scales(
        "Original seed_scale",
        params,
    )

    before_train = evaluate(
        model,
        predictors,
        train_loader,
        combo_table,
        device,
    )

    before_val = evaluate(
        model,
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\nBEFORE tuning"
    )

    print(
        f"  Train CE={before_train['ce']:.6f}, "
        f"Acc={before_train['accuracy']:.2f}%"
    )

    print(
        f"  Val   CE={before_val['ce']:.6f}, "
        f"Acc={before_val['accuracy']:.2f}%"
    )

    optimizer = torch.optim.AdamW(
        params,
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

    best_val_ce = float("inf")
    best_val_acc = -float("inf")
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
        model.eval()

        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = forward_dynamic_for_scale_training(
                model=model,
                predictors=predictors,
                images=images,
                combo_table=combo_table,
            )

            ce = F.cross_entropy(
                logits.float(),
                labels,
                reduction="mean",
            )

            reg = torch.zeros(
                (),
                dtype=ce.dtype,
                device=device,
            )

            if args.scale_reg_weight > 0:
                for parameter, original in zip(
                    params,
                    original_scales,
                ):
                    reg = reg + (
                        parameter
                        -
                        original
                    ).pow(2).mean()

                reg = (
                    reg
                    /
                    len(params)
                )

            loss = (
                ce
                +
                args.scale_reg_weight
                *
                reg
            )

            loss.backward()

            optimizer.step()

            clamp_scales(
                params,
                args.min_scale,
                args.max_scale,
            )

            B = labels.shape[0]

            train_loss_sum += (
                loss.item()
                *
                B
            )

            train_ce_sum += (
                ce.item()
                *
                B
            )

            train_correct += (
                logits.argmax(
                    dim=-1
                )
                ==
                labels
            ).sum().item()

            train_samples += B

        scheduler.step()

        val_metrics = evaluate(
            model,
            predictors,
            val_loader,
            combo_table,
            device,
        )

        train_loss = (
            train_loss_sum
            /
            train_samples
        )

        train_ce = (
            train_ce_sum
            /
            train_samples
        )

        train_acc = (
            100.0
            *
            train_correct
            /
            train_samples
        )

        current_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        scale_text = []

        for block_idx, parameter in enumerate(
            params
        ):
            values = (
                parameter
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            scale_text.append(
                f"B{block_idx}="
                f"{[round(v, 4) for v in values]}"
            )

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train loss={train_loss:.6f} | "
            f"train CE={train_ce:.6f} | "
            f"train acc={train_acc:.2f}% | "
            f"val CE={val_metrics['ce']:.6f} | "
            f"val acc={val_metrics['accuracy']:.2f}%"
        )

        print(
            "  "
            +
            " | ".join(
                scale_text
            )
        )

        better = (
            val_metrics[
                "ce"
            ]
            <
            best_val_ce
            -
            1e-12
        )

        if (
            not better
            and
            abs(
                val_metrics[
                    "ce"
                ]
                -
                best_val_ce
            )
            <=
            1e-12
            and
            val_metrics[
                "accuracy"
            ]
            >
            best_val_acc
        ):
            better = True

        if better:
            best_val_ce = (
                val_metrics[
                    "ce"
                ]
            )

            best_val_acc = (
                val_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            output = deepcopy(
                stage1_checkpoint
            )

            output[
                "model"
            ] = (
                model.state_dict()
            )

            output[
                "seed_scale_tuning"
            ] = {
                "best_epoch":
                    best_epoch,

                "best_val_ce":
                    best_val_ce,

                "best_val_accuracy":
                    best_val_acc,

                "source_stage1_checkpoint":
                    args.stage1_checkpoint,

                "predictor_checkpoint":
                    args.predictor_checkpoint,

                "scale_train_subset":
                    args.scale_train_subset,

                "scale_val_subset":
                    args.scale_val_subset,

                "scale_reg_weight":
                    args.scale_reg_weight,

                "min_scale":
                    args.min_scale,

                "max_scale":
                    args.max_scale,

                "learned_scales": [
                    p.detach().float().cpu()
                    for p in params
                ],
            }

            torch.save(
                output,
                args.output_checkpoint,
            )

    best = torch.load(
        args.output_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best[
            "model"
        ],
        strict=True,
    )

    final_names, final_params = (
        get_seed_scale_parameters(
            model
        )
    )

    after_train = evaluate(
        model,
        predictors,
        train_loader,
        combo_table,
        device,
    )

    after_val = evaluate(
        model,
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\n================ BEST SEED-SCALE CHECKPOINT ================"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Checkpoint: "
        f"{args.output_checkpoint}"
    )

    print_scales(
        "Best learned seed_scale",
        final_params,
    )

    print(
        "\nBEFORE -> AFTER"
    )

    print(
        f"Train CE: "
        f"{before_train['ce']:.6f}"
        " -> "
        f"{after_train['ce']:.6f}"
    )

    print(
        f"Train Acc: "
        f"{before_train['accuracy']:.2f}%"
        " -> "
        f"{after_train['accuracy']:.2f}%"
    )

    print(
        f"Val CE:   "
        f"{before_val['ce']:.6f}"
        " -> "
        f"{after_val['ce']:.6f}"
    )

    print(
        f"Val Acc:  "
        f"{before_val['accuracy']:.2f}%"
        " -> "
        f"{after_val['accuracy']:.2f}%"
    )

    print(
        "\nOnly seed_scale was optimized."
    )

    print(
        "Backbone, Mini/Main attention weights, Binder, "
        "and Utility+Interaction predictors were frozen."
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
        "\n이 실험은 official CIFAR-10 test를 사용하지 않습니다."
    )

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

    predictor_checkpoint = load_checkpoint(
        args.predictor_checkpoint,
        device,
        "Dynamic-state refined predictor checkpoint",
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
        checkpoint=predictor_checkpoint,
        model=model,
        feature_dim=feature_dim,
        device=device,
    )

    combinations, combo_table = (
        build_combo_table(
            model,
            device,
        )
    )

    print(
        "\nDirect pairs:"
    )

    print(
        torch.tensor(
            combinations,
            dtype=torch.long,
        )
    )

    train_set, val_set = (
        build_scale_datasets(
            args
        )
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # Separate loader with deterministic order for BEFORE/AFTER reporting.
    train_eval_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # Train loader is shuffled, but baseline reporting should use deterministic loader.
    # We pass train_eval_loader for baseline/final reporting and train_loader for updates.
    #
    # To keep the training function simple, temporarily attach the update loader.
    names, params = (
        get_seed_scale_parameters(
            model
        )
    )

    # Freeze before entering the custom loop below.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in params:
        parameter.requires_grad_(True)

    for predictor in predictors:
        predictor.eval()

    original_scales = [
        p.detach().clone()
        for p in params
    ]

    print(
        "\nTrainable parameters:"
    )

    total_trainable = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(
                f"  {name}: "
                f"{parameter.numel()}"
            )

            total_trainable += (
                parameter.numel()
            )

    print(
        f"Total trainable parameters: "
        f"{total_trainable}"
    )

    print_scales(
        "Original seed_scale",
        params,
    )

    before_train = evaluate(
        model,
        predictors,
        train_eval_loader,
        combo_table,
        device,
    )

    before_val = evaluate(
        model,
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\nBEFORE tuning"
    )

    print(
        f"  Train CE={before_train['ce']:.6f}, "
        f"Acc={before_train['accuracy']:.2f}%"
    )

    print(
        f"  Val   CE={before_val['ce']:.6f}, "
        f"Acc={before_val['accuracy']:.2f}%"
    )

    optimizer = torch.optim.AdamW(
        params,
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

    best_val_ce = float("inf")
    best_val_acc = -float("inf")
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
        model.eval()

        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = forward_dynamic_for_scale_training(
                model=model,
                predictors=predictors,
                images=images,
                combo_table=combo_table,
            )

            ce = F.cross_entropy(
                logits.float(),
                labels,
                reduction="mean",
            )

            reg = torch.zeros(
                (),
                dtype=ce.dtype,
                device=device,
            )

            if args.scale_reg_weight > 0:
                for parameter, original in zip(
                    params,
                    original_scales,
                ):
                    reg = (
                        reg
                        +
                        (
                            parameter
                            -
                            original
                        ).pow(2).mean()
                    )

                reg = (
                    reg
                    /
                    len(params)
                )

            loss = (
                ce
                +
                args.scale_reg_weight
                *
                reg
            )

            loss.backward()
            optimizer.step()

            clamp_scales(
                params,
                args.min_scale,
                args.max_scale,
            )

            B = labels.shape[0]

            train_loss_sum += (
                loss.item()
                *
                B
            )

            train_ce_sum += (
                ce.item()
                *
                B
            )

            train_correct += (
                logits.argmax(
                    dim=-1
                )
                ==
                labels
            ).sum().item()

            train_samples += B

        scheduler.step()

        val_metrics = evaluate(
            model,
            predictors,
            val_loader,
            combo_table,
            device,
        )

        train_loss = (
            train_loss_sum
            /
            train_samples
        )

        train_ce = (
            train_ce_sum
            /
            train_samples
        )

        train_acc = (
            100.0
            *
            train_correct
            /
            train_samples
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        scale_text = []

        for block_idx, parameter in enumerate(
            params
        ):
            values = (
                parameter
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            scale_text.append(
                f"B{block_idx}="
                f"{[round(v, 4) for v in values]}"
            )

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train loss={train_loss:.6f} | "
            f"train CE={train_ce:.6f} | "
            f"train acc={train_acc:.2f}% | "
            f"val CE={val_metrics['ce']:.6f} | "
            f"val acc={val_metrics['accuracy']:.2f}%"
        )

        print(
            "  "
            +
            " | ".join(
                scale_text
            )
        )

        better = (
            val_metrics[
                "ce"
            ]
            <
            best_val_ce
            -
            1e-12
        )

        if (
            not better
            and
            abs(
                val_metrics[
                    "ce"
                ]
                -
                best_val_ce
            )
            <=
            1e-12
            and
            val_metrics[
                "accuracy"
            ]
            >
            best_val_acc
        ):
            better = True

        if better:
            best_val_ce = (
                val_metrics[
                    "ce"
                ]
            )

            best_val_acc = (
                val_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            output = deepcopy(
                stage1
            )

            output[
                "model"
            ] = (
                model.state_dict()
            )

            output[
                "seed_scale_tuning"
            ] = {
                "best_epoch":
                    best_epoch,

                "best_val_ce":
                    best_val_ce,

                "best_val_accuracy":
                    best_val_acc,

                "source_stage1_checkpoint":
                    args.stage1_checkpoint,

                "predictor_checkpoint":
                    args.predictor_checkpoint,

                "learned_scales": [
                    p.detach()
                    .float()
                    .cpu()
                    for p in params
                ],
            }

            torch.save(
                output,
                args.output_checkpoint,
            )

    best = torch.load(
        args.output_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best[
            "model"
        ],
        strict=True,
    )

    _, best_params = (
        get_seed_scale_parameters(
            model
        )
    )

    after_train = evaluate(
        model,
        predictors,
        train_eval_loader,
        combo_table,
        device,
    )

    after_val = evaluate(
        model,
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print(
        "\n================ BEST SEED-SCALE CHECKPOINT ================"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Checkpoint: "
        f"{args.output_checkpoint}"
    )

    print_scales(
        "Best learned seed_scale",
        best_params,
    )

    print(
        "\nBEFORE -> AFTER"
    )

    print(
        f"Train CE: "
        f"{before_train['ce']:.6f}"
        " -> "
        f"{after_train['ce']:.6f}"
    )

    print(
        f"Train Acc: "
        f"{before_train['accuracy']:.2f}%"
        " -> "
        f"{after_train['accuracy']:.2f}%"
    )

    print(
        f"Val CE:   "
        f"{before_val['ce']:.6f}"
        " -> "
        f"{after_val['ce']:.6f}"
    )

    print(
        f"Val Acc:  "
        f"{before_val['accuracy']:.2f}%"
        " -> "
        f"{after_val['accuracy']:.2f}%"
    )

    print(
        "\nOnly 6 seed_scale values were optimized."
    )

    print(
        "All other model/predictor parameters were frozen."
    )


if __name__ == "__main__":
    main()
