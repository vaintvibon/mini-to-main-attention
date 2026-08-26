import argparse
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.counterfactual_direct_utility import CounterfactualDirectUtilityEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/content/cifar10")
    p.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage1_cifar10_balanced.pt",
    )
    p.add_argument(
        "--stage2-checkpoint",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_utility_predictor.pt",
    )
    p.add_argument(
        "--teacher-cache",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_counterfactual_teacher_cache.pt",
    )
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Must match Stage-1 split.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)

    # Stage-2 uses new, disjoint samples.
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)

    p.add_argument("--teacher-batch-size", type=int, default=32)
    p.add_argument("--utility-batch-size", type=int, default=128)
    p.add_argument("--utility-epochs", type=int, default=20)
    p.add_argument("--utility-lr", type=float, default=1e-3)
    p.add_argument("--utility-weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_stage1_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found:\n{path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise KeyError("Stage-1 checkpoint does not contain 'model'.")
    if "config" not in checkpoint:
        raise KeyError("Stage-1 checkpoint does not contain 'config'.")

    return checkpoint


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


def build_stage2_subsets(args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    total = len(base)

    required = (
        args.stage1_train_subset
        + args.stage1_val_subset
        + args.utility_train_subset
        + args.utility_val_subset
    )

    if required > total:
        raise ValueError(
            f"Requested splits exceed CIFAR-10 train set: {required} > {total}"
        )

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total, generator=g).tolist()

    offset = args.stage1_train_subset + args.stage1_val_subset

    utility_train_indices = perm[
        offset:
        offset + args.utility_train_subset
    ]

    offset += args.utility_train_subset

    utility_val_indices = perm[
        offset:
        offset + args.utility_val_subset
    ]

    train_set = Subset(base, utility_train_indices)
    val_set = Subset(base, utility_val_indices)

    print("\nStage-2 dataset split")
    print(f"Stage-1 train excluded: {args.stage1_train_subset}")
    print(f"Stage-1 val excluded: {args.stage1_val_subset}")
    print(f"Utility train: {len(train_set)}")
    print(f"Utility val: {len(val_set)}")

    return train_set, val_set


def build_reference_routing(evaluator, batch_size, depth, device):
    references = []

    for block_idx in range(depth):
        combo_idx = block_idx % evaluator.num_combinations

        combo = torch.tensor(
            evaluator.combinations[combo_idx],
            dtype=torch.long,
            device=device,
        )

        references.append(
            combo[None, :].expand(batch_size, -1).clone()
        )

    return references


@torch.no_grad()
def generate_teacher_split(
    model,
    dataset,
    evaluator,
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

    all_targets = []
    all_oracle = []
    all_subset_losses = []
    all_utilities = []

    seen = 0
    total = len(dataset)

    model.eval()

    print(f"\nGenerating counterfactual teacher: {split_name}")

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        result = evaluator.evaluate(x, labels)

        all_targets.append(result["teacher_target"].cpu())
        all_oracle.append(result["best_subset"].cpu())
        all_subset_losses.append(result["subset_losses"].cpu())
        all_utilities.append(result["head_utility"].cpu())

        combination_table = result["combination_table"].cpu()

        seen += x.shape[0]
        print(f"{split_name}: {seen}/{total}")

    return {
        "teacher_target": torch.cat(all_targets, dim=0),
        "oracle_best_subset": torch.cat(all_oracle, dim=0),
        "subset_losses": torch.cat(all_subset_losses, dim=0),
        "head_utility": torch.cat(all_utilities, dim=0),
        "combination_table": combination_table,
    }


def print_teacher_diagnostics(split_name, teacher_data):
    subset_losses = teacher_data["subset_losses"]
    head_utility = teacher_data["head_utility"]

    spread = (
        subset_losses.max(dim=-1).values
        - subset_losses.min(dim=-1).values
    )

    print(f"\n[{split_name}] Teacher diagnostics")
    print(
        f"Mean subset loss spread: "
        f"{spread.mean().item():.8e}"
    )
    print(
        f"Median subset loss spread: "
        f"{spread.median().item():.8e}"
    )
    print(
        f"Mean |head utility|: "
        f"{head_utility.abs().mean().item():.8e}"
    )


class UtilityTeacherDataset(Dataset):
    def __init__(self, image_dataset, teacher_data):
        if len(image_dataset) != teacher_data["teacher_target"].shape[0]:
            raise ValueError("Image / teacher size mismatch.")

        self.image_dataset = image_dataset
        self.teacher_target = teacher_data["teacher_target"]
        self.oracle_best_subset = teacher_data["oracle_best_subset"]
        self.subset_losses = teacher_data["subset_losses"]

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        x, label = self.image_dataset[index]

        return (
            x,
            label,
            self.teacher_target[index],
            self.oracle_best_subset[index],
            self.subset_losses[index],
        )


def freeze_backbone_unfreeze_utility(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    utility_parameters = []

    for block in model.blocks:
        predictor = block.attn.utility_predictor
        predictor.train()

        for parameter in predictor.parameters():
            parameter.requires_grad_(True)
            utility_parameters.append(parameter)

    return utility_parameters


def canonicalize_subset(subset):
    return subset.sort(dim=-1).values


def exact_pair_match(a, b):
    a = canonicalize_subset(a)
    b = canonicalize_subset(b)
    return (a == b).all(dim=-1).float()


def topk_overlap(a, b):
    matches = (
        a[..., :, None]
        ==
        b[..., None, :]
    )

    return matches.any(dim=-1).float().mean(dim=-1)


def subset_to_combo_indices(subset, combination_table):
    subset = canonicalize_subset(subset)
    combination_table = canonicalize_subset(combination_table)

    equality = (
        subset[:, :, None, :]
        ==
        combination_table[None, None, :, :]
    ).all(dim=-1)

    if not equality.any(dim=-1).all():
        raise RuntimeError(
            "Predicted pair missing from combination table."
        )

    return equality.float().argmax(dim=-1)


def forward_utility_logits(model, evaluator, x):
    references = build_reference_routing(
        evaluator=evaluator,
        batch_size=x.shape[0],
        depth=model.depth,
        device=x.device,
    )

    _, info_list = model(
        x,
        return_info=True,
        collect_taylor=False,
        forced_direct_indices_per_block=references,
        forced_uniform_mix=True,
    )

    return torch.stack(
        [info["utility_logits"] for info in info_list],
        dim=1,
    )


def compute_metrics(
    predicted_logits,
    teacher_target,
    oracle_subset,
    subset_losses,
    combination_table,
    direct_k,
):
    predicted_top1 = predicted_logits.argmax(dim=-1)
    teacher_top1 = teacher_target.argmax(dim=-1)

    top1 = (predicted_top1 == teacher_top1).float()

    predicted_pair = torch.topk(
        predicted_logits,
        k=direct_k,
        dim=-1,
    ).indices

    teacher_pair = torch.topk(
        teacher_target,
        k=direct_k,
        dim=-1,
    ).indices

    pred_teacher_exact = exact_pair_match(
        predicted_pair,
        teacher_pair,
    )

    pred_oracle_exact = exact_pair_match(
        predicted_pair,
        oracle_subset,
    )

    pred_oracle_overlap = topk_overlap(
        predicted_pair,
        oracle_subset,
    )

    combo_indices = subset_to_combo_indices(
        predicted_pair,
        combination_table,
    )

    predicted_loss = (
        subset_losses.gather(
            dim=-1,
            index=combo_indices[..., None],
        ).squeeze(-1)
    )

    oracle_loss = subset_losses.min(dim=-1).values
    regret = predicted_loss - oracle_loss

    return {
        "top1": top1,
        "pred_teacher_exact": pred_teacher_exact,
        "pred_oracle_exact": pred_oracle_exact,
        "pred_oracle_overlap": pred_oracle_overlap,
        "regret": regret,
        "predicted_pair": predicted_pair,
    }


@torch.no_grad()
def evaluate_utility_predictor(
    model,
    evaluator,
    loader,
    combination_table,
    device,
    direct_k,
):
    model.eval()

    total_kl = 0.0
    total_samples = 0

    all_top1 = []
    all_pred_teacher_exact = []
    all_pred_oracle_exact = []
    all_pred_oracle_overlap = []
    all_regret = []
    all_predicted_pairs = []

    combo_table_device = combination_table.to(device)

    for (
        x,
        _,
        teacher_target,
        oracle_subset,
        subset_losses,
    ) in loader:
        x = x.to(device, non_blocking=True)
        teacher_target = teacher_target.to(device, non_blocking=True)
        oracle_subset = oracle_subset.to(device, non_blocking=True)
        subset_losses = subset_losses.to(device, non_blocking=True)

        logits = forward_utility_logits(
            model,
            evaluator,
            x,
        )

        block_losses = []

        for block_idx in range(model.depth):
            block_loss = F.kl_div(
                F.log_softmax(
                    logits[:, block_idx, :],
                    dim=-1,
                ),
                teacher_target[:, block_idx, :],
                reduction="batchmean",
            )

            block_losses.append(block_loss)

        kl = torch.stack(block_losses).mean()
        batch_size = x.shape[0]

        total_kl += kl.item() * batch_size
        total_samples += batch_size

        metrics = compute_metrics(
            predicted_logits=logits,
            teacher_target=teacher_target,
            oracle_subset=oracle_subset,
            subset_losses=subset_losses,
            combination_table=combo_table_device,
            direct_k=direct_k,
        )

        all_top1.append(metrics["top1"].cpu())
        all_pred_teacher_exact.append(
            metrics["pred_teacher_exact"].cpu()
        )
        all_pred_oracle_exact.append(
            metrics["pred_oracle_exact"].cpu()
        )
        all_pred_oracle_overlap.append(
            metrics["pred_oracle_overlap"].cpu()
        )
        all_regret.append(metrics["regret"].cpu())
        all_predicted_pairs.append(
            metrics["predicted_pair"].cpu()
        )

    top1 = torch.cat(all_top1, dim=0)
    pred_teacher_exact = torch.cat(
        all_pred_teacher_exact,
        dim=0,
    )
    pred_oracle_exact = torch.cat(
        all_pred_oracle_exact,
        dim=0,
    )
    pred_oracle_overlap = torch.cat(
        all_pred_oracle_overlap,
        dim=0,
    )
    regret = torch.cat(all_regret, dim=0)
    predicted_pairs = torch.cat(
        all_predicted_pairs,
        dim=0,
    )

    return {
        "kl": total_kl / total_samples,
        "top1": top1.mean().item(),
        "pred_teacher_exact": pred_teacher_exact.mean().item(),
        "pred_oracle_exact": pred_oracle_exact.mean().item(),
        "pred_oracle_overlap": pred_oracle_overlap.mean().item(),
        "mean_regret": regret.mean().item(),
        "median_regret": regret.median().item(),
        "predicted_pairs": predicted_pairs,
    }


def print_metrics(metrics):
    print(f"KL: {metrics['kl']:.6f}")
    print(
        f"Top-1 teacher agreement: "
        f"{100.0 * metrics['top1']:.2f}%"
    )
    print(
        f"Pred pair vs Teacher exact: "
        f"{100.0 * metrics['pred_teacher_exact']:.2f}%"
    )
    print(
        f"Pred pair vs Oracle exact: "
        f"{100.0 * metrics['pred_oracle_exact']:.2f}%"
    )
    print(
        f"Pred pair vs Oracle overlap: "
        f"{100.0 * metrics['pred_oracle_overlap']:.2f}%"
    )
    print(
        f"Mean oracle regret: "
        f"{metrics['mean_regret']:.8e}"
    )
    print(
        f"Median oracle regret: "
        f"{metrics['median_regret']:.8e}"
    )


def train_utility_predictor(
    model,
    evaluator,
    train_loader,
    val_loader,
    combination_table,
    device,
    args,
):
    utility_parameters = freeze_backbone_unfreeze_utility(
        model
    )

    optimizer = torch.optim.AdamW(
        utility_parameters,
        lr=args.utility_lr,
        weight_decay=args.utility_weight_decay,
    )

    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.utility_epochs),
        )
    )

    print(
        "\n================ STAGE 2: "
        "UTILITY PREDICTOR TRAINING ================"
    )

    before = evaluate_utility_predictor(
        model=model,
        evaluator=evaluator,
        loader=val_loader,
        combination_table=combination_table,
        device=device,
        direct_k=model.direct_k,
    )

    print("\nValidation BEFORE training")
    print_metrics(before)

    best_val_kl = math.inf
    best_epoch = -1

    os.makedirs(
        os.path.dirname(args.stage2_checkpoint) or ".",
        exist_ok=True,
    )

    for epoch in range(1, args.utility_epochs + 1):
        model.eval()

        for block in model.blocks:
            block.attn.utility_predictor.train()

        running_loss = 0.0
        running_samples = 0

        for (
            x,
            _,
            teacher_target,
            _,
            _,
        ) in train_loader:
            x = x.to(device, non_blocking=True)
            teacher_target = teacher_target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = forward_utility_logits(
                model,
                evaluator,
                x,
            )

            block_losses = []

            for block_idx in range(model.depth):
                block_loss = F.kl_div(
                    F.log_softmax(
                        logits[:, block_idx, :],
                        dim=-1,
                    ),
                    teacher_target[:, block_idx, :],
                    reduction="batchmean",
                )

                block_losses.append(block_loss)

            loss = torch.stack(block_losses).mean()
            loss.backward()

            for parameter in utility_parameters:
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(
                            "Non-finite Utility Predictor gradient."
                        )

            optimizer.step()

            batch_size = x.shape[0]
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        train_kl = running_loss / running_samples

        val_metrics = evaluate_utility_predictor(
            model=model,
            evaluator=evaluator,
            loader=val_loader,
            combination_table=combination_table,
            device=device,
            direct_k=model.direct_k,
        )

        print(
            f"\nEpoch {epoch:02d}/{args.utility_epochs}"
        )
        print(f"Train KL: {train_kl:.6f}")
        print(f"Val KL: {val_metrics['kl']:.6f}")
        print(
            f"Val Top-1 teacher agreement: "
            f"{100.0 * val_metrics['top1']:.2f}%"
        )
        print(
            f"Val Pred pair vs Teacher exact: "
            f"{100.0 * val_metrics['pred_teacher_exact']:.2f}%"
        )
        print(
            f"Val Pred pair vs Oracle exact: "
            f"{100.0 * val_metrics['pred_oracle_exact']:.2f}%"
        )
        print(
            f"Val Pred pair vs Oracle overlap: "
            f"{100.0 * val_metrics['pred_oracle_overlap']:.2f}%"
        )
        print(
            f"Val mean oracle regret: "
            f"{val_metrics['mean_regret']:.8e}"
        )

        if val_metrics["kl"] < best_val_kl:
            best_val_kl = val_metrics["kl"]
            best_epoch = epoch

            utility_state = {}

            for block_idx, block in enumerate(model.blocks):
                predictor = block.attn.utility_predictor

                utility_state[f"block_{block_idx}"] = {
                    key: value.detach().cpu()
                    for key, value in predictor.state_dict().items()
                }

            torch.save(
                {
                    "utility_predictors": utility_state,
                    "best_epoch": best_epoch,
                    "best_val_kl": best_val_kl,
                    "stage1_checkpoint": args.stage1_checkpoint,
                    "config": vars(args),
                },
                args.stage2_checkpoint,
            )

        lr_scheduler.step()

    print("\nBest utility checkpoint:")
    print(args.stage2_checkpoint)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val KL: {best_val_kl:.6f}")

    best = torch.load(
        args.stage2_checkpoint,
        map_location=device,
        weights_only=False,
    )

    for block_idx, block in enumerate(model.blocks):
        block.attn.utility_predictor.load_state_dict(
            best["utility_predictors"][f"block_{block_idx}"]
        )

    after = evaluate_utility_predictor(
        model=model,
        evaluator=evaluator,
        loader=val_loader,
        combination_table=combination_table,
        device=device,
        direct_k=model.direct_k,
    )

    print(
        "\n================ FINAL HELD-OUT VALIDATION ================"
    )
    print_metrics(after)

    return before, after


def print_predicted_pair_frequency(
    predicted_pairs,
    combinations,
):
    print(
        "\nPredicted pair frequency "
        "(held-out validation)"
    )

    depth = predicted_pairs.shape[1]

    for block_idx in range(depth):
        pairs = canonicalize_subset(
            predicted_pairs[:, block_idx, :]
        ).tolist()

        counter = Counter(
            tuple(int(v) for v in pair)
            for pair in pairs
        )

        total = sum(counter.values())

        print(f"\nBlock {block_idx}:")

        for combo in combinations:
            count = counter.get(tuple(combo), 0)
            percentage = 100.0 * count / total

            print(
                f"  {tuple(combo)}: "
                f"{count:4d} "
                f"({percentage:6.2f}%)"
            )


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

    checkpoint = load_stage1_checkpoint(
        args.stage1_checkpoint,
        device,
    )

    model = build_model_from_stage1_config(
        checkpoint["config"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    print("\nLoaded Stage-1 checkpoint:")
    print(args.stage1_checkpoint)

    print(
        f"\nModel: "
        f"depth={model.depth}, "
        f"mini_heads={model.mini_heads}, "
        f"main_heads={model.main_heads}, "
        f"direct_k={model.direct_k}"
    )

    utility_train_set, utility_val_set = (
        build_stage2_subsets(args)
    )

    evaluator = CounterfactualDirectUtilityEvaluator(
        model=model,
        mini_heads=model.mini_heads,
        direct_k=model.direct_k,
        target_temperature=1.0,
    )

    if (
        os.path.exists(args.teacher_cache)
        and
        not args.rebuild_cache
    ):
        print("\nLoading cached counterfactual teacher:")
        print(args.teacher_cache)

        teacher_cache = torch.load(
            args.teacher_cache,
            map_location="cpu",
            weights_only=False,
        )

        train_teacher = teacher_cache["train"]
        val_teacher = teacher_cache["val"]

    else:
        train_teacher = generate_teacher_split(
            model=model,
            dataset=utility_train_set,
            evaluator=evaluator,
            device=device,
            batch_size=args.teacher_batch_size,
            num_workers=args.num_workers,
            split_name="utility-train",
        )

        val_teacher = generate_teacher_split(
            model=model,
            dataset=utility_val_set,
            evaluator=evaluator,
            device=device,
            batch_size=args.teacher_batch_size,
            num_workers=args.num_workers,
            split_name="utility-val",
        )

        os.makedirs(
            os.path.dirname(args.teacher_cache) or ".",
            exist_ok=True,
        )

        torch.save(
            {
                "train": train_teacher,
                "val": val_teacher,
                "stage1_checkpoint": args.stage1_checkpoint,
                "split_config": {
                    "seed": args.seed,
                    "stage1_train_subset": args.stage1_train_subset,
                    "stage1_val_subset": args.stage1_val_subset,
                    "utility_train_subset": args.utility_train_subset,
                    "utility_val_subset": args.utility_val_subset,
                },
            },
            args.teacher_cache,
        )

        print("\nSaved teacher cache:")
        print(args.teacher_cache)

    print_teacher_diagnostics(
        "utility-train",
        train_teacher,
    )

    print_teacher_diagnostics(
        "utility-val",
        val_teacher,
    )

    train_dataset = UtilityTeacherDataset(
        utility_train_set,
        train_teacher,
    )

    val_dataset = UtilityTeacherDataset(
        utility_val_set,
        val_teacher,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.utility_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.utility_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    combination_table = train_teacher["combination_table"]

    before, after = train_utility_predictor(
        model=model,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        combination_table=combination_table,
        device=device,
        args=args,
    )

    print_predicted_pair_frequency(
        after["predicted_pairs"],
        evaluator.combinations,
    )

    print("\n================ STAGE-2 SUMMARY ================")

    print("Held-out Top-1 teacher agreement:")
    print(
        f"{100.0 * before['top1']:.2f}%"
        " -> "
        f"{100.0 * after['top1']:.2f}%"
    )

    print("\nHeld-out Pred pair vs Oracle exact:")
    print(
        f"{100.0 * before['pred_oracle_exact']:.2f}%"
        " -> "
        f"{100.0 * after['pred_oracle_exact']:.2f}%"
    )

    print("\nHeld-out Pred pair vs Oracle overlap:")
    print(
        f"{100.0 * before['pred_oracle_overlap']:.2f}%"
        " -> "
        f"{100.0 * after['pred_oracle_overlap']:.2f}%"
    )

    print("\nHeld-out mean oracle regret:")
    print(
        f"{before['mean_regret']:.8e}"
        " -> "
        f"{after['mean_regret']:.8e}"
    )

    print(
        "\nStage-2 CIFAR-10 Utility Predictor "
        "training completed."
    )


if __name__ == "__main__":
    main()