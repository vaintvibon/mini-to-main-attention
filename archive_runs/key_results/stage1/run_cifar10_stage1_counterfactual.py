import argparse
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.balanced_direct_scheduler import BalancedDirectSubsetScheduler
from models.counterfactual_direct_utility import CounterfactualDirectUtilityEvaluator


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=str, default="/content/cifar10")
    p.add_argument("--checkpoint", type=str, default="/content/stage1_cifar10_balanced.pt")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-subset", type=int, default=4096)
    p.add_argument("--val-subset", type=int, default=1000)

    p.add_argument("--cf-samples", type=int, default=200)
    p.add_argument("--cf-batch-size", type=int, default=16)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--main-heads", type=int, default=3)
    p.add_argument("--mini-heads", type=int, default=4)
    p.add_argument("--mini-head-dim", type=int, default=16)
    p.add_argument("--direct-k", type=int, default=2)

    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    eval_base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    total = len(train_base)

    if args.train_subset + args.val_subset > total:
        raise ValueError(
            f"train_subset + val_subset exceeds CIFAR-10 train size: "
            f"{args.train_subset} + {args.val_subset} > {total}"
        )

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total, generator=g).tolist()

    train_indices = perm[:args.train_subset]
    val_indices = perm[
        args.train_subset:
        args.train_subset + args.val_subset
    ]

    train_set = Subset(train_base, train_indices)
    val_set = Subset(eval_base, val_indices)

    return train_set, val_set


def build_model(args):
    return DynamicMiniMainViT(
        img_size=32,
        patch_size=4,
        num_classes=10,

        embed_dim=args.embed_dim,
        depth=args.depth,

        main_heads=args.main_heads,

        mini_heads=args.mini_heads,
        mini_head_dim=args.mini_head_dim,
        pool_ratio=2,

        utility_hidden_dim=64,

        direct_k=args.direct_k,
        mix_temperature=1.0,

        bind_dim=64,
        bind_temperature=1.0,

        mlp_ratio=4.0,

        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


def freeze_utility_predictor(model):
    for block in model.blocks:
        predictor = block.attn.utility_predictor
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)


def assert_utility_frozen(model):
    for block_idx, block in enumerate(model.blocks):
        predictor = block.attn.utility_predictor
        for name, parameter in predictor.named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(
                    f"Utility Predictor is not frozen: "
                    f"block={block_idx}, parameter={name}"
                )


@torch.no_grad()
def evaluate_reference(
    model,
    loader,
    scheduler,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    eval_step = 0

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        forced = scheduler.get_for_all_blocks(
            batch_size=x.shape[0],
            depth=model.depth,
            step=eval_step,
            device=device,
        )

        logits = model(
            x,
            return_info=False,
            forced_direct_indices_per_block=forced,
            forced_uniform_mix=True,
        )

        loss_sum = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        total_loss += loss_sum.item()
        total_correct += (
            logits.argmax(dim=-1) == labels
        ).sum().item()
        total_samples += labels.numel()

        eval_step += 1

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
    )


def train_stage1(
    model,
    train_loader,
    val_loader,
    scheduler,
    device,
    args,
):
    freeze_utility_predictor(model)
    assert_utility_frozen(model)

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(
        1,
        args.epochs * len(train_loader),
    )

    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    combo_to_id = {
        tuple(combo): idx
        for idx, combo in enumerate(
            scheduler.combinations
        )
    }

    coverage = torch.zeros(
        model.depth,
        scheduler.num_combinations,
        dtype=torch.long,
    )

    global_step = 0

    print("\n================ STAGE 1: BALANCED WARM-UP ================")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"direct combinations: {scheduler.combinations}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            forced = scheduler.get_for_all_blocks(
                batch_size=x.shape[0],
                depth=model.depth,
                step=global_step,
                device=device,
            )

            for block_idx in range(model.depth):
                for row in forced[block_idx].detach().cpu().tolist():
                    coverage[
                        block_idx,
                        combo_to_id[tuple(row)],
                    ] += 1

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    x,
                    return_info=False,
                    forced_direct_indices_per_block=forced,
                    forced_uniform_mix=True,
                )

                loss = F.cross_entropy(
                    logits,
                    labels,
                )

            scaler.scale(loss).backward()

            # Utility Predictor must remain outside Stage-1 optimization.
            for block_idx, block in enumerate(model.blocks):
                for name, parameter in (
                    block.attn.utility_predictor.named_parameters()
                ):
                    if parameter.grad is not None:
                        raise RuntimeError(
                            "Gradient leaked into Utility Predictor: "
                            f"block={block_idx}, parameter={name}"
                        )

            scaler.step(optimizer)
            scaler.update()
            scheduler_lr.step()

            running_loss += loss.item() * labels.numel()
            running_correct += (
                logits.argmax(dim=-1) == labels
            ).sum().item()
            running_samples += labels.numel()

            global_step += 1

        train_loss = running_loss / running_samples
        train_acc = 100.0 * running_correct / running_samples

        val_loss, val_acc = evaluate_reference(
            model,
            val_loader,
            scheduler,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_acc:.2f}% | "
            f"val loss={val_loss:.4f} "
            f"acc={val_acc:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

    print("\nBalanced routing coverage:")
    print(coverage)

    os.makedirs(
        os.path.dirname(args.checkpoint) or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "coverage": coverage,
            "direct_combinations": scheduler.combinations,
        },
        args.checkpoint,
    )

    print(f"\nSaved Stage-1 checkpoint:\n{args.checkpoint}")

    return coverage


def canonicalize_subset(subset):
    return subset.sort(dim=-1).values


def exact_pair_match(a, b):
    a = canonicalize_subset(a)
    b = canonicalize_subset(b)

    return (
        (a == b)
        .all(dim=-1)
        .float()
    )


def topk_overlap(a, b):
    matches = (
        a[..., :, None]
        ==
        b[..., None, :]
    )

    return (
        matches
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )


def subset_to_combo_indices(
    subset,
    combo_table,
):
    subset = canonicalize_subset(subset)
    combo_table = canonicalize_subset(combo_table)

    equality = (
        subset[:, :, None, :]
        ==
        combo_table[None, None, :, :]
    ).all(dim=-1)

    if not equality.any(dim=-1).all():
        raise RuntimeError(
            "A selected pair is missing from the combination table."
        )

    return equality.float().argmax(dim=-1)


def format_pair(pair):
    return tuple(int(v) for v in pair)


@torch.no_grad()
def counterfactual_validation(
    model,
    val_loader,
    device,
    args,
):
    print("\n================ COUNTERFACTUAL VALIDATION ================")

    model.eval()

    evaluator = CounterfactualDirectUtilityEvaluator(
        model=model,
        mini_heads=args.mini_heads,
        direct_k=args.direct_k,
        target_temperature=1.0,
    )

    all_subset_losses = []
    all_head_utility = []
    all_teacher_target = []
    all_best_subset = []

    seen = 0

    for x, labels in val_loader:
        if seen >= args.cf_samples:
            break

        remaining = args.cf_samples - seen

        if x.shape[0] > remaining:
            x = x[:remaining]
            labels = labels[:remaining]

        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        result = evaluator.evaluate(
            x,
            labels,
        )

        all_subset_losses.append(
            result["subset_losses"].cpu()
        )

        all_head_utility.append(
            result["head_utility"].cpu()
        )

        all_teacher_target.append(
            result["teacher_target"].cpu()
        )

        all_best_subset.append(
            result["best_subset"].cpu()
        )

        combination_table = (
            result["combination_table"].cpu()
        )

        seen += x.shape[0]

        print(
            f"counterfactual evaluated: "
            f"{seen}/{args.cf_samples}"
        )

    subset_losses = torch.cat(
        all_subset_losses,
        dim=0,
    )

    head_utility = torch.cat(
        all_head_utility,
        dim=0,
    )

    teacher_target = torch.cat(
        all_teacher_target,
        dim=0,
    )

    oracle_best_subset = torch.cat(
        all_best_subset,
        dim=0,
    )

    subset_spread = (
        subset_losses.max(dim=-1).values
        -
        subset_losses.min(dim=-1).values
    )

    teacher_topk = torch.topk(
        teacher_target,
        k=args.direct_k,
        dim=-1,
    ).indices

    exact = exact_pair_match(
        teacher_topk,
        oracle_best_subset,
    )

    overlap = topk_overlap(
        teacher_topk,
        oracle_best_subset,
    )

    teacher_combo_indices = (
        subset_to_combo_indices(
            teacher_topk,
            combination_table,
        )
    )

    teacher_loss = (
        subset_losses.gather(
            dim=-1,
            index=teacher_combo_indices[..., None],
        )
        .squeeze(-1)
    )

    oracle_loss = (
        subset_losses.min(dim=-1).values
    )

    regret = (
        teacher_loss
        -
        oracle_loss
    )

    print("\n---------------- SIGNAL MAGNITUDE ----------------")
    print(
        f"Mean subset loss spread: "
        f"{subset_spread.mean().item():.8e}"
    )
    print(
        f"Median subset loss spread: "
        f"{subset_spread.median().item():.8e}"
    )
    print(
        f"Mean |head utility|: "
        f"{head_utility.abs().mean().item():.8e}"
    )

    print("\n---------------- TEACHER vs ORACLE ----------------")
    print(
        f"Exact pair match: "
        f"{100.0 * exact.mean().item():.2f}%"
    )
    print(
        f"Top-{args.direct_k} overlap: "
        f"{100.0 * overlap.mean().item():.2f}%"
    )
    print(
        f"Mean oracle regret: "
        f"{regret.mean().item():.8e}"
    )
    print(
        f"Median oracle regret: "
        f"{regret.median().item():.8e}"
    )

    print("\n---------------- ORACLE PAIR FREQUENCY ----------------")

    # Block-wise
    for block_idx in range(model.depth):
        counter = Counter(
            format_pair(pair)
            for pair in canonicalize_subset(
                oracle_best_subset[:, block_idx, :]
            ).tolist()
        )

        total = sum(counter.values())

        print(f"\nBlock {block_idx}:")
        for combo in evaluator.combinations:
            count = counter.get(
                tuple(combo),
                0,
            )

            pct = (
                100.0 * count / total
                if total > 0
                else 0.0
            )

            print(
                f"  {tuple(combo)}: "
                f"{count:4d} "
                f"({pct:6.2f}%)"
            )

    # Overall
    overall_counter = Counter(
        format_pair(pair)
        for pair in canonicalize_subset(
            oracle_best_subset
        ).reshape(
            -1,
            args.direct_k,
        ).tolist()
    )

    overall_total = sum(
        overall_counter.values()
    )

    print("\nOverall:")
    for combo in evaluator.combinations:
        count = overall_counter.get(
            tuple(combo),
            0,
        )

        pct = (
            100.0 * count / overall_total
            if overall_total > 0
            else 0.0
        )

        print(
            f"  {tuple(combo)}: "
            f"{count:4d} "
            f"({pct:6.2f}%)"
        )

    dominant_count = max(
        overall_counter.values()
    )

    dominant_fraction = (
        dominant_count
        /
        overall_total
    )

    probs = torch.tensor(
        [
            overall_counter.get(
                tuple(combo),
                0,
            )
            /
            overall_total

            for combo in evaluator.combinations
        ],
        dtype=torch.float32,
    )

    nz = probs > 0

    entropy = (
        -(
            probs[nz]
            *
            probs[nz].log()
        ).sum()
    )

    normalized_entropy = (
        entropy
        /
        math.log(
            len(
                evaluator.combinations
            )
        )
    )

    print("\n---------------- ORACLE DIVERSITY ----------------")
    print(
        f"Dominant pair fraction: "
        f"{100.0 * dominant_fraction:.2f}%"
    )
    print(
        f"Normalized pair entropy: "
        f"{normalized_entropy.item():.4f}"
    )

    print("\nInterpretation guide:")
    print(
        "- Larger subset-loss spread / |utility|: "
        "Direct choice matters more."
    )
    print(
        "- Lower dominant-pair fraction + higher entropy: "
        "best Direct pair changes more across inputs."
    )
    print(
        "- High teacher-vs-oracle match + low regret: "
        "per-head utility Top-K is a good approximation."
    )

    return {
        "subset_losses": subset_losses,
        "head_utility": head_utility,
        "teacher_target": teacher_target,
        "oracle_best_subset": oracle_best_subset,
        "subset_spread": subset_spread,
        "teacher_oracle_exact": exact,
        "teacher_oracle_overlap": overlap,
        "teacher_oracle_regret": regret,
    }


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

    train_set, val_set = build_datasets(args)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    cf_loader = DataLoader(
        val_set,
        batch_size=args.cf_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args).to(device)

    balanced_scheduler = BalancedDirectSubsetScheduler(
        mini_heads=args.mini_heads,
        direct_k=args.direct_k,
    )

    train_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=balanced_scheduler,
        device=device,
        args=args,
    )

    counterfactual_validation(
        model=model,
        val_loader=cf_loader,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()