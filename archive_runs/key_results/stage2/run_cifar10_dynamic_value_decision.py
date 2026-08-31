import argparse
import itertools
import math
import os
import random
from collections import Counter
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
        "--predictor-checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # All data regions already consumed in previous experiments.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)
    p.add_argument("--diagnostic-subset", type=int, default=1000)
    p.add_argument("--scale-train-subset", type=int, default=1000)
    p.add_argument("--scale-val-subset", type=int, default=500)
    p.add_argument("--previous-heldout-subset", type=int, default=1000)

    # Static baseline is selected only on an already-consumed training region.
    # It never sees the fresh decision-set labels.
    p.add_argument("--static-search-samples", type=int, default=1000)

    # Completely fresh decision set begins AFTER previous heldout.
    p.add_argument("--decision-samples", type=int, default=1000)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--random-trials", type=int, default=5)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)

    p.add_argument("--amp", action="store_true")

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


def load_model(checkpoint_path, device):
    ckpt = load_checkpoint(
        checkpoint_path,
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

    return model, ckpt


def load_predictors(
    checkpoint_path,
    model,
    device,
):
    ckpt = load_checkpoint(
        checkpoint_path,
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

    for block_idx, predictor in enumerate(predictors):
        predictor.load_state_dict(
            states[f"block_{block_idx}"],
            strict=True,
        )

        predictor.eval()

        for p in predictor.parameters():
            p.requires_grad_(False)

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
            transforms.Normalize(mean, std),
        ]
    )


def build_datasets(args):
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

    # Static baseline search:
    # reuse the prior seed-scale training region [7596, 8596) by default.
    static_start = (
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

    static_end = (
        static_start
        +
        args.static_search_samples
    )

    # Fresh decision split:
    # 4096 + 1000 + 1000 + 500 + 1000 + 1000 + 500 + 1000 = 10096
    decision_start = (
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
    )

    decision_end = (
        decision_start
        +
        args.decision_samples
    )

    if static_end > len(base):
        raise ValueError(
            f"Static search split ends at {static_end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    if decision_end > len(base):
        raise ValueError(
            f"Decision split ends at {decision_end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    static_indices = permutation[
        static_start:
        static_end
    ]

    decision_indices = permutation[
        decision_start:
        decision_end
    ]

    if set(static_indices) & set(decision_indices):
        raise RuntimeError(
            "Static-search and decision splits overlap."
        )

    print("\nData split:")
    print(
        f"  Static-search offset: "
        f"[{static_start}, {static_end})"
    )
    print(
        f"  Fresh decision offset: "
        f"[{decision_start}, {decision_end})"
    )
    print(
        "  official CIFAR-10 test is NOT used."
    )

    return (
        Subset(base, static_indices),
        Subset(base, decision_indices),
    )


# ============================================================
# Routing helpers
# ============================================================

def amp_context(device, use_amp):
    if use_amp and device.type == "cuda":
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

    combo_indices = list(
        range(len(combinations))
    )

    route_configs = list(
        itertools.product(
            combo_indices,
            repeat=model.depth,
        )
    )

    return combinations, route_configs


def combo_indices_to_forced(
    route_config,
    combinations,
    batch_size,
    device,
):
    forced = []

    for combo_idx in route_config:
        pair = torch.tensor(
            combinations[int(combo_idx)],
            dtype=torch.long,
            device=device,
        )

        forced.append(
            pair[None, :]
            .expand(batch_size, -1)
            .clone()
        )

    return forced


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

    combo_device = combo_table.to(
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

        forced_pair = combo_device[
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
# Static search
# ============================================================

@torch.no_grad()
def search_best_static_route(
    model,
    loader,
    combinations,
    route_configs,
    device,
    use_amp,
):
    model.eval()

    total_losses = torch.zeros(
        len(route_configs),
        dtype=torch.float64,
    )

    total_samples = 0

    print(
        "\n================ STATIC ROUTE SEARCH ================"
    )

    print(
        f"Candidate whole-network routes: "
        f"{len(route_configs)}"
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
            forced = combo_indices_to_forced(
                route_config=route_config,
                combinations=combinations,
                batch_size=B,
                device=device,
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

            total_losses[
                route_idx
            ] += losses.sum().item()

        total_samples += B

        print(
            f"Static search: "
            f"{total_samples}/"
            f"{len(loader.dataset)}"
        )

    mean_losses = (
        total_losses
        /
        total_samples
    )

    best_idx = int(
        mean_losses.argmin().item()
    )

    best_route = route_configs[
        best_idx
    ]

    print("\nBest fixed route:")

    for block_idx, combo_idx in enumerate(
        best_route
    ):
        print(
            f"  Block {block_idx}: "
            f"{combinations[combo_idx]}"
        )

    print(
        f"Static-search mean CE: "
        f"{mean_losses[best_idx].item():.6f}"
    )

    return best_route


# ============================================================
# Decision-set evaluation
# ============================================================

@torch.no_grad()
def evaluate_dynamic(
    model,
    predictors,
    loader,
    combo_table,
    device,
    use_amp,
):
    losses_all = []
    correct_all = []
    selected_all = []

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
            logits, selected = forward_dynamic(
                model=model,
                predictors=predictors,
                images=images,
                combo_table=combo_table,
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

        losses_all.append(
            losses.cpu()
        )

        correct_all.append(
            correct.cpu()
        )

        selected_all.append(
            selected.cpu()
        )

    return {
        "losses":
            torch.cat(losses_all),

        "correct":
            torch.cat(correct_all),

        "selected":
            torch.cat(selected_all),
    }


@torch.no_grad()
def evaluate_static(
    model,
    loader,
    combinations,
    route_config,
    device,
    use_amp,
):
    losses_all = []
    correct_all = []

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

        forced = combo_indices_to_forced(
            route_config=route_config,
            combinations=combinations,
            batch_size=B,
            device=device,
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

        losses_all.append(
            losses.cpu()
        )

        correct_all.append(
            correct.cpu()
        )

    return {
        "losses":
            torch.cat(losses_all),

        "correct":
            torch.cat(correct_all),
    }


@torch.no_grad()
def evaluate_random(
    model,
    loader,
    combinations,
    device,
    use_amp,
    trials,
    seed,
):
    trial_results = []

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
        device=device,
    )

    for trial in range(trials):
        g = torch.Generator().manual_seed(
            seed + 1000 + trial
        )

        losses_all = []
        correct_all = []

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

            forced = []

            for _ in range(model.depth):
                idx = torch.randint(
                    0,
                    len(combinations),
                    (B,),
                    generator=g,
                ).to(device)

                forced.append(
                    combo_table[idx]
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

            losses_all.append(
                losses.cpu()
            )

            correct_all.append(
                correct.cpu()
            )

        result = {
            "losses":
                torch.cat(losses_all),

            "correct":
                torch.cat(correct_all),
        }

        trial_results.append(
            result
        )

        print(
            f"Random trial {trial + 1}/{trials}: "
            f"CE={result['losses'].mean().item():.6f}, "
            f"Acc={100.0 * result['correct'].float().mean().item():.2f}%"
        )

    mean_losses = torch.stack(
        [
            r["losses"]
            for r in trial_results
        ],
        dim=0,
    ).mean(dim=0)

    # Majority here is not used. For paired accuracy comparison,
    # use the mean correctness probability across random trials.
    mean_correct = torch.stack(
        [
            r["correct"].float()
            for r in trial_results
        ],
        dim=0,
    ).mean(dim=0)

    return {
        "losses":
            mean_losses,

        "correct_prob":
            mean_correct,

        "trial_results":
            trial_results,
    }


@torch.no_grad()
def evaluate_oracle(
    model,
    loader,
    combinations,
    route_configs,
    device,
    use_amp,
):
    """
    Diagnostic upper bound:
    for each sample, enumerate all 6^depth whole-network routes
    and choose minimum CE using the label.
    """

    best_losses_all = []
    best_correct_all = []
    best_route_all = []

    seen = 0

    print(
        "\n================ GLOBAL ORACLE ================"
    )

    print(
        f"Routes per sample: "
        f"{len(route_configs)}"
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

        best_loss = torch.full(
            (B,),
            float("inf"),
            device=device,
        )

        best_correct = torch.zeros(
            B,
            dtype=torch.bool,
            device=device,
        )

        best_route = torch.zeros(
            B,
            dtype=torch.long,
            device=device,
        )

        for route_idx, route_config in enumerate(
            route_configs
        ):
            forced = combo_indices_to_forced(
                route_config=route_config,
                combinations=combinations,
                batch_size=B,
                device=device,
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

            better = losses < best_loss

            best_loss = torch.where(
                better,
                losses,
                best_loss,
            )

            best_correct = torch.where(
                better,
                correct,
                best_correct,
            )

            best_route = torch.where(
                better,
                torch.full_like(
                    best_route,
                    route_idx,
                ),
                best_route,
            )

        best_losses_all.append(
            best_loss.cpu()
        )

        best_correct_all.append(
            best_correct.cpu()
        )

        best_route_all.append(
            best_route.cpu()
        )

        seen += B

        print(
            f"Oracle: "
            f"{seen}/"
            f"{len(loader.dataset)}"
        )

    return {
        "losses":
            torch.cat(best_losses_all),

        "correct":
            torch.cat(best_correct_all),

        "route":
            torch.cat(best_route_all),
    }


# ============================================================
# Paired bootstrap
# ============================================================

def bootstrap_mean_ci(
    values,
    repeats,
    seed,
):
    values = values.float().cpu()

    n = values.numel()

    g = torch.Generator().manual_seed(seed)

    means = torch.empty(
        repeats,
        dtype=torch.float32,
    )

    chunk = 250
    done = 0

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


def compare_paired(
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

        "wrong_to_correct":
            (
                (~reference["correct"])
                &
                candidate["correct"]
            ).sum().item(),

        "correct_to_wrong":
            (
                reference["correct"]
                &
                (~candidate["correct"])
            ).sum().item(),
    }


# ============================================================
# Reporting
# ============================================================

def mean_ce(result):
    return result["losses"].mean().item()


def accuracy(result):
    return (
        100.0
        *
        result["correct"].float().mean().item()
    )


def print_pair_frequency(
    selected,
    combinations,
):
    print(
        "\nDynamic Direct pair frequency"
    )

    for block_idx in range(
        selected.shape[1]
    ):
        print(
            f"\nBlock {block_idx}:"
        )

        counter = Counter(
            int(v)
            for v in selected[
                :,
                block_idx
            ].tolist()
        )

        total = selected.shape[0]

        for combo_idx, combo in enumerate(
            combinations
        ):
            count = counter.get(
                combo_idx,
                0,
            )

            print(
                f"  {combo}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
            )


def print_comparison(
    title,
    stats,
):
    print(
        f"\n{title}"
    )

    lo, hi = stats["ce_ci"]

    print(
        f"  ΔCE(candidate-reference): "
        f"{stats['delta_ce']:+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.8f}, {hi:+.8f}]"
    )

    lo, hi = stats["acc_ci"]

    print(
        f"  ΔAccuracy: "
        f"{stats['delta_acc']:+.3f}%p"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.3f}, {hi:+.3f}]%p"
    )

    print(
        f"  Wrong -> Correct: "
        f"{stats['wrong_to_correct']}"
    )

    print(
        f"  Correct -> Wrong: "
        f"{stats['correct_to_wrong']}"
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    seed_everything(args.seed)

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

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("AMP:", use_amp)

    print(
        "\n이 실험은 현재 핵심 가설의 판정용입니다."
    )

    print(
        "Fresh decision set과 official CIFAR-10 test는 "
        "모델/route 선택에 사용하지 않습니다."
    )

    model, _ = load_model(
        args.backbone_checkpoint,
        device,
    )

    predictors = load_predictors(
        args.predictor_checkpoint,
        model,
        device,
    )

    combinations, route_configs = (
        build_route_space(model)
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    print(
        "\nDirect pair table:"
    )

    print(combo_table)

    print(
        f"Whole-network route configurations: "
        f"{len(route_configs)}"
    )

    static_set, decision_set = (
        build_datasets(args)
    )

    static_loader = DataLoader(
        static_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    decision_loader = DataLoader(
        decision_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Static route chosen without fresh decision labels.
    # --------------------------------------------------------

    best_static_route = search_best_static_route(
        model=model,
        loader=static_loader,
        combinations=combinations,
        route_configs=route_configs,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # Fresh decision set.
    # --------------------------------------------------------

    print(
        "\n================ FRESH DECISION SET ================"
    )

    random_result = evaluate_random(
        model=model,
        loader=decision_loader,
        combinations=combinations,
        device=device,
        use_amp=use_amp,
        trials=args.random_trials,
        seed=args.seed,
    )

    print(
        "\nEvaluating best fixed route..."
    )

    static_result = evaluate_static(
        model=model,
        loader=decision_loader,
        combinations=combinations,
        route_config=best_static_route,
        device=device,
        use_amp=use_amp,
    )

    print(
        "Evaluating Dynamic Utility + Interaction..."
    )

    dynamic_result = evaluate_dynamic(
        model=model,
        predictors=predictors,
        loader=decision_loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    oracle_result = evaluate_oracle(
        model=model,
        loader=decision_loader,
        combinations=combinations,
        route_configs=route_configs,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    random_ce_trials = torch.tensor(
        [
            r["losses"].mean().item()
            for r in random_result[
                "trial_results"
            ]
        ]
    )

    random_acc_trials = torch.tensor(
        [
            100.0
            *
            r["correct"].float().mean().item()
            for r in random_result[
                "trial_results"
            ]
        ]
    )

    print(
        "\n================ DECISIVE SUMMARY ================"
    )

    print(
        "\nRandom"
    )

    print(
        f"  CE: "
        f"{random_ce_trials.mean().item():.6f} "
        f"± {random_ce_trials.std(unbiased=False).item():.6f}"
    )

    print(
        f"  Accuracy: "
        f"{random_acc_trials.mean().item():.2f}% "
        f"± {random_acc_trials.std(unbiased=False).item():.2f}%"
    )

    print(
        "\nStatic"
    )

    for block_idx, combo_idx in enumerate(
        best_static_route
    ):
        print(
            f"  Block {block_idx}: "
            f"{combinations[combo_idx]}"
        )

    print(
        f"  CE: "
        f"{mean_ce(static_result):.6f}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy(static_result):.2f}%"
    )

    print(
        "\nDynamic Utility + Interaction"
    )

    print(
        f"  CE: "
        f"{mean_ce(dynamic_result):.6f}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy(dynamic_result):.2f}%"
    )

    print(
        "\nOracle"
    )

    print(
        f"  CE: "
        f"{mean_ce(oracle_result):.6f}"
    )

    print(
        f"  Accuracy: "
        f"{accuracy(oracle_result):.2f}%"
    )

    # --------------------------------------------------------
    # Core paired comparisons.
    # --------------------------------------------------------

    dynamic_vs_static = compare_paired(
        reference=static_result,
        candidate=dynamic_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 100,
    )

    oracle_vs_static = compare_paired(
        reference=static_result,
        candidate=oracle_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 200,
    )

    oracle_vs_dynamic = compare_paired(
        reference=dynamic_result,
        candidate=oracle_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 300,
    )

    print_comparison(
        "STATIC -> DYNAMIC",
        dynamic_vs_static,
    )

    print_comparison(
        "STATIC -> ORACLE",
        oracle_vs_static,
    )

    print_comparison(
        "DYNAMIC -> ORACLE",
        oracle_vs_dynamic,
    )

    # --------------------------------------------------------
    # How much of the available oracle CE gap does Dynamic close?
    # --------------------------------------------------------

    static_ce = mean_ce(
        static_result
    )

    dynamic_ce = mean_ce(
        dynamic_result
    )

    oracle_ce = mean_ce(
        oracle_result
    )

    available_gap = (
        static_ce
        -
        oracle_ce
    )

    captured_gap = (
        static_ce
        -
        dynamic_ce
    )

    if available_gap > 0:
        gap_closure = (
            100.0
            *
            captured_gap
            /
            available_gap
        )
    else:
        gap_closure = float("nan")

    print(
        "\n================ ORACLE GAP ================"
    )

    print(
        f"Static - Oracle CE gap: "
        f"{available_gap:.8f}"
    )

    print(
        f"Static - Dynamic CE gain: "
        f"{captured_gap:.8f}"
    )

    print(
        f"Dynamic captured oracle gap: "
        f"{gap_closure:.2f}%"
    )

    print_pair_frequency(
        dynamic_result[
            "selected"
        ],
        combinations,
    )

    # --------------------------------------------------------
    # Explicit research verdict.
    # --------------------------------------------------------

    ds_ce_lo, ds_ce_hi = (
        dynamic_vs_static[
            "ce_ci"
        ]
    )

    so_ce_lo, so_ce_hi = (
        oracle_vs_static[
            "ce_ci"
        ]
    )

    dynamic_clear_ce_win = (
        dynamic_vs_static[
            "delta_ce"
        ]
        <
        0
        and
        ds_ce_hi
        <
        0
    )

    oracle_clear_value = (
        oracle_vs_static[
            "delta_ce"
        ]
        <
        0
        and
        so_ce_hi
        <
        0
    )

    print(
        "\n================ RESEARCH VERDICT ================"
    )

    if (
        dynamic_clear_ce_win
        and
        oracle_clear_value
    ):
        print(
            "GO: 입력별 Dynamic Direct 선택이 고정 선택보다 "
            "fresh held-out CE에서 명확히 우수합니다."
        )

        print(
            "현재 핵심 가설을 유지하고 더 큰 모델/데이터셋으로 확장할 근거가 있습니다."
        )

    elif (
        (not dynamic_clear_ce_win)
        and
        oracle_clear_value
    ):
        print(
            "PREDICTOR BOTTLENECK: 입력별 더 좋은 route는 실제로 존재하지만, "
            "현재 Predictor가 그 이득을 충분히 회수하지 못합니다."
        )

        print(
            "Dynamic-routing 연구 질문은 살아 있고, 다음 병목은 선택 예측입니다."
        )

    else:
        print(
            "PIVOT WARNING: Oracle조차 고정 route보다 명확한 이득이 없습니다."
        )

        print(
            "현재 backbone에서 입력별 Dynamic Direct 선택의 실효성이 약합니다."
        )


if __name__ == "__main__":
    main()
