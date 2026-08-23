import argparse
import os

import torch
from torch.utils.data import DataLoader

from models.set_aware_mini_head_utility import (
    SetAwareMiniHeadUtility,
)

from models.counterfactual_direct_utility import (
    CounterfactualDirectUtilityEvaluator,
)

from run_cifar10_stage2_utility import (
    load_stage1_checkpoint,
    build_model_from_stage1_config,
    build_stage2_subsets,
    UtilityTeacherDataset,
    train_utility_predictor,
    print_predicted_pair_frequency,
    seed_everything,
)


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
        "--teacher-cache",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_counterfactual_teacher_cache.pt"
        ),
    )

    p.add_argument(
        "--stage2-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "stage2_setaware_utility_predictor.pt"
        ),
    )

    # Must match cached split.
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

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
        "--utility-batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--utility-epochs",
        type=int,
        default=20,
    )

    p.add_argument(
        "--utility-lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--utility-weight-decay",
        type=float,
        default=1e-4,
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
        "--num-workers",
        type=int,
        default=2,
    )

    # Compatibility placeholders used by imported trainer config.
    p.add_argument(
        "--teacher-batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    return p.parse_args()


def replace_with_setaware_predictors(
    model,
    hidden_dim,
    dropout,
    device,
):
    print(
        "\nReplacing local-only Utility Predictors "
        "with Set-Aware predictors."
    )

    for block_idx, block in enumerate(
        model.blocks
    ):
        attn = (
            block.attn
        )

        predictor = (
            SetAwareMiniHeadUtility(
                mini_head_dim=(
                    attn.mini_head_dim
                ),
                hidden_dim=hidden_dim,
                dropout=dropout,
                has_cls_token=True,
            )
            .to(device)
        )

        attn.utility_predictor = (
            predictor
        )

        parameter_count = sum(
            p.numel()
            for p in predictor.parameters()
        )

        print(
            f"Block {block_idx}: "
            f"{parameter_count:,} predictor parameters"
        )


def validate_cache_split(
    cache,
    args,
):
    if "split_config" not in cache:
        print(
            "\nWARNING: cache has no split_config; "
            "cannot verify split metadata."
        )
        return

    expected = {
        "seed":
            args.seed,

        "stage1_train_subset":
            args.stage1_train_subset,

        "stage1_val_subset":
            args.stage1_val_subset,

        "utility_train_subset":
            args.utility_train_subset,

        "utility_val_subset":
            args.utility_val_subset,
    }

    actual = (
        cache[
            "split_config"
        ]
    )

    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                "Teacher cache split mismatch: "
                f"{key}: cache={actual.get(key)}, "
                f"requested={value}"
            )


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

    # =========================================================
    # 1. Load old Stage-1 architecture/checkpoint FIRST.
    #
    # This preserves strict compatibility with Stage-1.
    # =========================================================

    checkpoint = (
        load_stage1_checkpoint(
            args.stage1_checkpoint,
            device,
        )
    )

    model = (
        build_model_from_stage1_config(
            checkpoint[
                "config"
            ]
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ],
        strict=True,
    )

    print(
        "\nLoaded Stage-1 backbone:"
    )

    print(
        args.stage1_checkpoint
    )

    # =========================================================
    # 2. Replace ONLY Utility Predictor.
    #
    # Backbone/Mini/Main/Binder weights stay exactly Stage-1.
    # =========================================================

    replace_with_setaware_predictors(
        model=model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device,
    )

    # =========================================================
    # 3. Use EXACT SAME cached counterfactual teachers.
    # =========================================================

    if not os.path.exists(
        args.teacher_cache
    ):
        raise FileNotFoundError(
            "Teacher cache not found:\n"
            f"{args.teacher_cache}\n"
            "Run Stage-2 teacher generation first."
        )

    cache = torch.load(
        args.teacher_cache,
        map_location="cpu",
        weights_only=False,
    )

    validate_cache_split(
        cache,
        args,
    )

    train_teacher = (
        cache[
            "train"
        ]
    )

    val_teacher = (
        cache[
            "val"
        ]
    )

    # =========================================================
    # 4. Rebuild identical image splits.
    # =========================================================

    (
        utility_train_set,
        utility_val_set,
    ) = build_stage2_subsets(
        args
    )

    train_dataset = (
        UtilityTeacherDataset(
            utility_train_set,
            train_teacher,
        )
    )

    val_dataset = (
        UtilityTeacherDataset(
            utility_val_set,
            val_teacher,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.utility_batch_size
        ),
        shuffle=True,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=(
            args.utility_batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    evaluator = (
        CounterfactualDirectUtilityEvaluator(
            model=model,
            mini_heads=model.mini_heads,
            direct_k=model.direct_k,
            target_temperature=1.0,
        )
    )

    combination_table = (
        train_teacher[
            "combination_table"
        ]
    )

    # =========================================================
    # 5. Same trainer / same KL / same split / same teacher.
    #
    # Therefore the only intended experimental change is:
    #
    # local-only Utility Predictor
    #           ->
    # Set-Aware Utility Predictor
    # =========================================================

    before, after = (
        train_utility_predictor(
            model=model,
            evaluator=evaluator,
            train_loader=train_loader,
            val_loader=val_loader,
            combination_table=(
                combination_table
            ),
            device=device,
            args=args,
        )
    )

    print_predicted_pair_frequency(
        after[
            "predicted_pairs"
        ],
        evaluator.combinations,
    )

    print(
        "\n================ SET-AWARE COMPARISON ================"
    )

    print(
        "Held-out Top-1 teacher agreement:"
    )

    print(
        f"{100.0 * before['top1']:.2f}%"
        " -> "
        f"{100.0 * after['top1']:.2f}%"
    )

    print(
        "\nHeld-out Pred pair vs Teacher exact:"
    )

    print(
        f"{100.0 * before['pred_teacher_exact']:.2f}%"
        " -> "
        f"{100.0 * after['pred_teacher_exact']:.2f}%"
    )

    print(
        "\nHeld-out Pred pair vs Oracle exact:"
    )

    print(
        f"{100.0 * before['pred_oracle_exact']:.2f}%"
        " -> "
        f"{100.0 * after['pred_oracle_exact']:.2f}%"
    )

    print(
        "\nHeld-out Pred pair vs Oracle overlap:"
    )

    print(
        f"{100.0 * before['pred_oracle_overlap']:.2f}%"
        " -> "
        f"{100.0 * after['pred_oracle_overlap']:.2f}%"
    )

    print(
        "\nHeld-out mean oracle regret:"
    )

    print(
        f"{before['mean_regret']:.8e}"
        " -> "
        f"{after['mean_regret']:.8e}"
    )

    print(
        "\nReference baselines from previous diagnostic:"
    )

    print(
        "Random exact pair: 16.67%"
    )

    print(
        "Random overlap: 50.00%"
    )

    print(
        "Global-prior exact: 20.90%"
    )

    print(
        "Global-prior overlap: 53.55%"
    )

    print(
        "Teacher exact: 71.20%"
    )

    print(
        "Teacher overlap: 85.60%"
    )

    print(
        "\nSet-Aware Utility Predictor experiment completed."
    )


if __name__ == "__main__":
    main()
