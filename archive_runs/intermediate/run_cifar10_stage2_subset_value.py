import argparse
import itertools
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.mini_subset_value_predictor import MiniSubsetValuePredictor

from run_cifar10_stage2_utility import (
    load_stage1_checkpoint,
    build_model_from_stage1_config,
    build_stage2_subsets,
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
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage1_cifar10_balanced.pt",
    )
    p.add_argument(
        "--teacher-cache",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_counterfactual_teacher_cache.pt",
    )
    p.add_argument(
        "--feature-cache",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_subset_feature_cache.pt",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_subset_value_predictor.pt",
    )

    p.add_argument("--rebuild-feature-cache", action="store_true")

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)

    p.add_argument("--feature-batch-size", type=int, default=128)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Direct subset supervision.
    p.add_argument("--regret-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--target-temperature", type=float, default=0.5)

    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


def validate_teacher_cache(cache, args):
    if "split_config" not in cache:
        print("WARNING: teacher cache has no split_config metadata.")
        return

    expected = {
        "seed": args.seed,
        "stage1_train_subset": args.stage1_train_subset,
        "stage1_val_subset": args.stage1_val_subset,
        "utility_train_subset": args.utility_train_subset,
        "utility_val_subset": args.utility_val_subset,
    }

    actual = cache["split_config"]

    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"Teacher cache split mismatch: {key}, "
                f"cache={actual.get(key)}, requested={value}"
            )


def build_reference_routing(
    combinations,
    batch_size,
    depth,
    device,
):
    refs = []

    for block_idx in range(depth):
        combo = torch.tensor(
            combinations[block_idx % len(combinations)],
            dtype=torch.long,
            device=device,
        )

        refs.append(
            combo[None, :].expand(batch_size, -1).clone()
        )

    return refs


@torch.no_grad()
def extract_features(
    model,
    dataset,
    prototype,
    combinations,
    device,
    batch_size,
    num_workers,
    split_name,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    outputs = []
    model.eval()

    seen = 0
    total = len(dataset)

    print(f"\nExtracting Mini features: {split_name}")

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        refs = build_reference_routing(
            combinations=combinations,
            batch_size=x.shape[0],
            depth=model.depth,
            device=device,
        )

        _, info_list = model(
            x,
            return_info=True,
            collect_taylor=False,
            forced_direct_indices_per_block=refs,
            forced_uniform_mix=True,
        )

        block_features = []

        for info in info_list:
            block_features.append(
                prototype.extract_local_features(
                    info["mini_contexts"],
                    info["mini_attn"],
                ).cpu()
            )

        outputs.append(
            torch.stack(block_features, dim=1)
        )

        seen += x.shape[0]
        print(f"{split_name}: {seen}/{total}")

    return torch.cat(outputs, dim=0)


def load_or_build_feature_cache(
    model,
    train_set,
    val_set,
    prototype,
    combinations,
    device,
    args,
):
    if (
        os.path.exists(args.feature_cache)
        and
        not args.rebuild_feature_cache
    ):
        print("\nLoading feature cache:")
        print(args.feature_cache)

        cache = torch.load(
            args.feature_cache,
            map_location="cpu",
            weights_only=False,
        )

        return cache["train"], cache["val"]

    train_features = extract_features(
        model=model,
        dataset=train_set,
        prototype=prototype,
        combinations=combinations,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        split_name="subset-train",
    )

    val_features = extract_features(
        model=model,
        dataset=val_set,
        prototype=prototype,
        combinations=combinations,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        split_name="subset-val",
    )

    os.makedirs(
        os.path.dirname(args.feature_cache) or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "train": train_features,
            "val": val_features,
        },
        args.feature_cache,
    )

    print("\nSaved feature cache:")
    print(args.feature_cache)

    return train_features, val_features


class OfflineDataset(Dataset):
    def __init__(self, features, subset_losses):
        if features.shape[0] != subset_losses.shape[0]:
            raise ValueError("Feature/teacher sample count mismatch.")

        self.features = features.float()
        self.subset_losses = subset_losses.float()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.subset_losses[idx],
        )


def forward_predictors(
    predictors,
    features,
):
    return torch.stack(
        [
            predictors[b].forward_from_features(
                features[:, b, :, :]
            )
            for b in range(features.shape[1])
        ],
        dim=1,
    )


def training_loss(
    scores,
    subset_losses,
    target_temperature,
    regret_weight,
    kl_weight,
    eps=1e-8,
):
    min_loss = subset_losses.min(
        dim=-1,
        keepdim=True,
    ).values

    max_loss = subset_losses.max(
        dim=-1,
        keepdim=True,
    ).values

    spread = (
        max_loss - min_loss
    ).clamp_min(eps)

    normalized_cost = (
        subset_losses - min_loss
    ) / spread

    pred_probs = torch.softmax(
        scores,
        dim=-1,
    )

    expected_regret = (
        pred_probs * normalized_cost
    ).sum(dim=-1).mean()

    target_probs = torch.softmax(
        -normalized_cost / target_temperature,
        dim=-1,
    )

    kl = F.kl_div(
        F.log_softmax(scores, dim=-1),
        target_probs,
        reduction="batchmean",
    )

    total = (
        regret_weight * expected_regret
        +
        kl_weight * kl
    )

    return total, expected_regret.detach(), kl.detach()


def head_overlap(
    pred_pair,
    oracle_pair,
):
    match = (
        pred_pair[..., :, None]
        ==
        oracle_pair[..., None, :]
    )

    return (
        match.any(dim=-1)
        .float()
        .mean(dim=-1)
    )


@torch.no_grad()
def evaluate(
    predictors,
    loader,
    combo_table,
    device,
):
    for predictor in predictors:
        predictor.eval()

    exact_all = []
    overlap_all = []
    regret_all = []
    pred_idx_all = []

    combo_table_device = combo_table.to(device)

    for features, subset_losses in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        subset_losses = subset_losses.to(
            device,
            non_blocking=True,
        )

        scores = forward_predictors(
            predictors,
            features,
        )

        pred_idx = scores.argmax(dim=-1)
        oracle_idx = subset_losses.argmin(dim=-1)

        exact = (
            pred_idx == oracle_idx
        ).float()

        pred_pair = combo_table_device[pred_idx]
        oracle_pair = combo_table_device[oracle_idx]

        overlap = head_overlap(
            pred_pair,
            oracle_pair,
        )

        selected_loss = (
            subset_losses.gather(
                dim=-1,
                index=pred_idx[..., None],
            ).squeeze(-1)
        )

        oracle_loss = subset_losses.min(dim=-1).values

        regret = selected_loss - oracle_loss

        exact_all.append(exact.cpu())
        overlap_all.append(overlap.cpu())
        regret_all.append(regret.cpu())
        pred_idx_all.append(pred_idx.cpu())

    exact = torch.cat(exact_all, dim=0)
    overlap = torch.cat(overlap_all, dim=0)
    regret = torch.cat(regret_all, dim=0)
    pred_idx = torch.cat(pred_idx_all, dim=0)

    return {
        "exact": exact.mean().item(),
        "overlap": overlap.mean().item(),
        "mean_regret": regret.mean().item(),
        "median_regret": regret.median().item(),
        "pred_idx": pred_idx,
    }


def evaluate_best_static(
    train_losses,
    val_losses,
    combo_table,
):
    # One fixed subset per block, chosen only from TRAIN.
    static_idx = (
        train_losses.mean(dim=0)
        .argmin(dim=-1)
    )

    B = val_losses.shape[0]

    static_batch = (
        static_idx[None, :]
        .expand(B, -1)
    )

    oracle_idx = val_losses.argmin(dim=-1)

    exact = (
        static_batch == oracle_idx
    ).float()

    pred_pair = combo_table[static_batch]
    oracle_pair = combo_table[oracle_idx]

    overlap = head_overlap(
        pred_pair,
        oracle_pair,
    )

    selected_loss = (
        val_losses.gather(
            dim=-1,
            index=static_batch[..., None],
        ).squeeze(-1)
    )

    oracle_loss = val_losses.min(dim=-1).values

    regret = selected_loss - oracle_loss

    return {
        "pair": combo_table[static_idx],
        "exact": exact.mean().item(),
        "overlap": overlap.mean().item(),
        "mean_regret": regret.mean().item(),
        "median_regret": regret.median().item(),
    }


def print_metrics(title, m):
    print(f"\n{title}")
    print(
        f"Oracle exact pair: "
        f"{100.0 * m['exact']:.2f}%"
    )
    print(
        f"Oracle Top-2 overlap: "
        f"{100.0 * m['overlap']:.2f}%"
    )
    print(
        f"Mean oracle regret: "
        f"{m['mean_regret']:.8e}"
    )
    print(
        f"Median oracle regret: "
        f"{m['median_regret']:.8e}"
    )


def print_frequency(pred_idx, combo_table):
    print("\nPredicted subset frequency (held-out validation)")

    for block_idx in range(pred_idx.shape[1]):
        counter = Counter(
            pred_idx[:, block_idx].tolist()
        )

        total = pred_idx.shape[0]

        print(f"\nBlock {block_idx}:")

        for i, combo in enumerate(combo_table.tolist()):
            count = counter.get(i, 0)
            print(
                f"  {tuple(combo)}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
            )


def train(
    predictors,
    train_loader,
    val_loader,
    combo_table,
    device,
    args,
):
    params = [
        p
        for predictor in predictors
        for p in predictor.parameters()
    ]

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    before = evaluate(
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print_metrics(
        "Validation BEFORE training",
        before,
    )

    best_regret = math.inf
    best_exact = -1.0
    best_epoch = -1

    os.makedirs(
        os.path.dirname(args.checkpoint) or ".",
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):
        for predictor in predictors:
            predictor.train()

        total_loss = 0.0
        total_nr = 0.0
        total_kl = 0.0
        total_n = 0

        for features, subset_losses in train_loader:
            features = features.to(
                device,
                non_blocking=True,
            )

            subset_losses = subset_losses.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            scores = forward_predictors(
                predictors,
                features,
            )

            loss, nr, kl = training_loss(
                scores=scores,
                subset_losses=subset_losses,
                target_temperature=args.target_temperature,
                regret_weight=args.regret_weight,
                kl_weight=args.kl_weight,
            )

            loss.backward()
            optimizer.step()

            B = features.shape[0]

            total_loss += loss.item() * B
            total_nr += nr.item() * B
            total_kl += kl.item() * B
            total_n += B

        lr_scheduler.step()

        val = evaluate(
            predictors,
            val_loader,
            combo_table,
            device,
        )

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"train={total_loss / total_n:.6f} | "
            f"norm-regret={total_nr / total_n:.6f} | "
            f"KL={total_kl / total_n:.6f}"
        )

        print(
            f"Val exact={100.0 * val['exact']:.2f}% | "
            f"overlap={100.0 * val['overlap']:.2f}% | "
            f"regret={val['mean_regret']:.8e}"
        )

        better = (
            val["mean_regret"] < best_regret - 1e-12
        )

        if (
            not better
            and
            abs(val["mean_regret"] - best_regret) <= 1e-12
            and
            val["exact"] > best_exact
        ):
            better = True

        if better:
            best_regret = val["mean_regret"]
            best_exact = val["exact"]
            best_epoch = epoch

            torch.save(
                {
                    "predictors": {
                        f"block_{i}": predictor.state_dict()
                        for i, predictor in enumerate(predictors)
                    },
                    "best_epoch": best_epoch,
                    "best_val_regret": best_regret,
                    "best_val_exact": best_exact,
                    "combinations": combo_table,
                    "config": vars(args),
                },
                args.checkpoint,
            )

    best = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    for i, predictor in enumerate(predictors):
        predictor.load_state_dict(
            best["predictors"][f"block_{i}"]
        )

    after = evaluate(
        predictors,
        val_loader,
        combo_table,
        device,
    )

    print("\nBest checkpoint:")
    print(args.checkpoint)
    print(f"Best epoch: {best_epoch}")
    print_metrics(
        "FINAL HELD-OUT subset-value validation",
        after,
    )

    return before, after


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    stage1 = load_stage1_checkpoint(
        args.stage1_checkpoint,
        device,
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

    print("\nLoaded and frozen Stage-1 backbone:")
    print(args.stage1_checkpoint)

    teacher_cache = torch.load(
        args.teacher_cache,
        map_location="cpu",
        weights_only=False,
    )

    validate_teacher_cache(
        teacher_cache,
        args,
    )

    train_losses = teacher_cache[
        "train"
    ][
        "subset_losses"
    ].float()

    val_losses = teacher_cache[
        "val"
    ][
        "subset_losses"
    ].float()

    combo_table = teacher_cache[
        "train"
    ][
        "combination_table"
    ].long()

    expected_combos = list(
        itertools.combinations(
            range(model.mini_heads),
            model.direct_k,
        )
    )

    expected_table = torch.tensor(
        expected_combos,
        dtype=torch.long,
    )

    if not torch.equal(
        combo_table,
        expected_table,
    ):
        raise ValueError(
            "Combination table/order mismatch."
        )

    print("\nDirect subset table:")
    print(combo_table)

    static = evaluate_best_static(
        train_losses,
        val_losses,
        combo_table,
    )

    print("\n================ BEST STATIC BASELINE ================")
    print("Fixed pair per block:")
    print(static["pair"])
    print_metrics(
        "Static baseline on held-out validation",
        static,
    )

    train_set, val_set = build_stage2_subsets(args)

    prototype = MiniSubsetValuePredictor(
        mini_head_dim=model.blocks[0].attn.mini_head_dim,
        mini_heads=model.mini_heads,
        direct_k=model.direct_k,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_features, val_features = (
        load_or_build_feature_cache(
            model=model,
            train_set=train_set,
            val_set=val_set,
            prototype=prototype,
            combinations=expected_combos,
            device=device,
            args=args,
        )
    )

    print("\nFeature shapes:")
    print("train:", tuple(train_features.shape))
    print("val:", tuple(val_features.shape))

    predictors = torch.nn.ModuleList(
        [
            MiniSubsetValuePredictor(
                mini_head_dim=model.blocks[i].attn.mini_head_dim,
                mini_heads=model.mini_heads,
                direct_k=model.direct_k,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )
            for i in range(model.depth)
        ]
    ).to(device)

    for i, predictor in enumerate(predictors):
        print(
            f"Block {i} predictor params: "
            f"{sum(p.numel() for p in predictor.parameters()):,}"
        )

    train_ds = OfflineDataset(
        train_features,
        train_losses,
    )

    val_ds = OfflineDataset(
        val_features,
        val_losses,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    before, after = train(
        predictors=predictors,
        train_loader=train_loader,
        val_loader=val_loader,
        combo_table=combo_table,
        device=device,
        args=args,
    )

    print_frequency(
        after["pred_idx"],
        combo_table,
    )

    print("\n================ SUBSET-VALUE SUMMARY ================")
    print(
        f"Random exact expectation: "
        f"{100.0 / combo_table.shape[0]:.2f}%"
    )
    print(
        f"Random overlap expectation: "
        f"{100.0 * model.direct_k / model.mini_heads:.2f}%"
    )

    print("\nBest static:")
    print(
        f"Exact={100.0 * static['exact']:.2f}% | "
        f"Overlap={100.0 * static['overlap']:.2f}% | "
        f"Regret={static['mean_regret']:.8e}"
    )

    print("\nSubset predictor BEFORE -> AFTER:")
    print(
        f"Exact: "
        f"{100.0 * before['exact']:.2f}%"
        " -> "
        f"{100.0 * after['exact']:.2f}%"
    )
    print(
        f"Overlap: "
        f"{100.0 * before['overlap']:.2f}%"
        " -> "
        f"{100.0 * after['overlap']:.2f}%"
    )
    print(
        f"Regret: "
        f"{before['mean_regret']:.8e}"
        " -> "
        f"{after['mean_regret']:.8e}"
    )

    print(
        "\nThis experiment predicts the six Direct subsets directly; "
        "no per-head marginal utility compression is used."
    )


if __name__ == "__main__":
    main()
