import argparse
import itertools
import math
import os
import random
from dataclasses import dataclass

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
        "--old-backbone",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage1_cifar10_balanced.pt"
        ),
    )

    p.add_argument(
        "--old-predictor",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_predictor.pt"
        ),
    )

    p.add_argument(
        "--new-backbone",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage1_cifar10_seedscale_tuned.pt"
        ),
    )

    p.add_argument(
        "--new-predictor",
        type=str,
        default=(
            "/content/drive/MyDrive/mini-to-main-attention/checkpoints/"
            "stage2_dynamic_state_refined_after_seedscale.pt"
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    # All previously consumed CIFAR-10 train-set regions.
    p.add_argument("--stage1-train-subset", type=int, default=4096)
    p.add_argument("--stage1-val-subset", type=int, default=1000)
    p.add_argument("--utility-train-subset", type=int, default=1000)
    p.add_argument("--utility-val-subset", type=int, default=500)
    p.add_argument("--diagnostic-subset", type=int, default=1000)
    p.add_argument("--scale-train-subset", type=int, default=1000)
    p.add_argument("--scale-val-subset", type=int, default=500)

    # Fresh held-out set begins after every region above.
    p.add_argument("--heldout-samples", type=int, default=1000)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    # Paired bootstrap over the SAME samples.
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


def build_heldout_dataset(args):
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
    )

    end = start + args.heldout_samples

    if end > len(base):
        raise ValueError(
            f"Held-out split ends at {end}, "
            f"but CIFAR-10 train has {len(base)} samples."
        )

    print("\nFresh held-out split:")
    print(f"  permutation offset: [{start}, {end})")
    print(f"  samples: {args.heldout_samples}")
    print("  official CIFAR-10 test is NOT used.")

    return Subset(
        base,
        permutation[start:end],
    )


# ============================================================
# Dynamic inference
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
def dynamic_forward(
    model,
    predictors,
    images,
    combo_table,
):
    """
    Sequential actual dynamic routing:
      current block state
      -> current predictor
      -> pair choice
      -> block execution
      -> next state

    Remaining Mini heads use uniform Mix,
    matching the controlled training/evaluation setup.
    """

    x = prepare_tokens(
        model,
        images,
    )

    combo_table = combo_table.to(
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

        forced_pair = combo_table[
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
# Evaluation
# ============================================================

@dataclass
class EvalResult:
    name: str
    losses: torch.Tensor
    correct: torch.Tensor
    predictions: torch.Tensor
    selected: torch.Tensor

    @property
    def ce(self):
        return self.losses.mean().item()

    @property
    def accuracy(self):
        return (
            100.0
            *
            self.correct.float().mean().item()
        )


@torch.no_grad()
def evaluate_system(
    name,
    model,
    predictors,
    loader,
    combo_table,
    device,
    use_amp,
):
    model.eval()

    for predictor in predictors:
        predictor.eval()

    losses_all = []
    correct_all = []
    pred_all = []
    selected_all = []

    seen = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if (
            use_amp
            and
            device.type == "cuda"
        ):
            ctx = torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            )
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            logits, selected = dynamic_forward(
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

        predictions = logits.argmax(
            dim=-1
        )

        correct = (
            predictions
            ==
            labels
        )

        losses_all.append(
            losses.cpu()
        )

        correct_all.append(
            correct.cpu()
        )

        pred_all.append(
            predictions.cpu()
        )

        selected_all.append(
            selected.cpu()
        )

        seen += labels.shape[0]

        print(
            f"{name}: "
            f"{seen}/{len(loader.dataset)}"
        )

    return EvalResult(
        name=name,
        losses=torch.cat(losses_all),
        correct=torch.cat(correct_all),
        predictions=torch.cat(pred_all),
        selected=torch.cat(selected_all),
    )


# ============================================================
# Paired comparison
# ============================================================

def paired_bootstrap_ci(
    delta,
    repeats,
    seed,
):
    """
    delta: per-sample candidate - reference metric.
    """

    delta = delta.float().cpu()

    n = delta.numel()

    g = torch.Generator().manual_seed(
        seed
    )

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

        indices = torch.randint(
            low=0,
            high=n,
            size=(r, n),
            generator=g,
        )

        means[
            done:
            done + r
        ] = (
            delta[
                indices
            ].mean(
                dim=1
            )
        )

        done += r

    q = torch.quantile(
        means,
        torch.tensor(
            [
                0.025,
                0.975,
            ]
        ),
    )

    return (
        q[0].item(),
        q[1].item(),
    )


def compare_systems(
    reference,
    candidate,
    repeats,
    seed,
):
    loss_delta = (
        candidate.losses
        -
        reference.losses
    )

    # accuracy contribution per sample in percentage points
    acc_delta = (
        candidate.correct.float()
        -
        reference.correct.float()
    ) * 100.0

    loss_ci = paired_bootstrap_ci(
        loss_delta,
        repeats,
        seed,
    )

    acc_ci = paired_bootstrap_ci(
        acc_delta,
        repeats,
        seed + 1,
    )

    improved = (
        (~reference.correct)
        &
        candidate.correct
    ).sum().item()

    worsened = (
        reference.correct
        &
        (~candidate.correct)
    ).sum().item()

    unchanged_correct = (
        reference.correct
        &
        candidate.correct
    ).sum().item()

    unchanged_wrong = (
        (~reference.correct)
        &
        (~candidate.correct)
    ).sum().item()

    route_disagreement = (
        candidate.selected
        !=
        reference.selected
    ).float().mean(
        dim=0
    )

    return {
        "delta_ce":
            loss_delta.mean().item(),

        "delta_ce_ci":
            loss_ci,

        "delta_acc_pp":
            acc_delta.mean().item(),

        "delta_acc_ci":
            acc_ci,

        "improved":
            improved,

        "worsened":
            worsened,

        "unchanged_correct":
            unchanged_correct,

        "unchanged_wrong":
            unchanged_wrong,

        "route_disagreement":
            route_disagreement,
    }


def print_result(result):
    print(
        f"\n{result.name}"
    )

    print(
        f"  CE: "
        f"{result.ce:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{result.accuracy:.2f}%"
    )


def print_comparison(
    title,
    stats,
):
    print(
        f"\n{title}"
    )

    lo, hi = stats[
        "delta_ce_ci"
    ]

    print(
        f"  ΔCE(candidate-reference): "
        f"{stats['delta_ce']:+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.8f}, {hi:+.8f}]"
    )

    lo, hi = stats[
        "delta_acc_ci"
    ]

    print(
        f"  ΔAccuracy: "
        f"{stats['delta_acc_pp']:+.3f}%p"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{lo:+.3f}, {hi:+.3f}]%p"
    )

    print(
        f"  Wrong -> Correct: "
        f"{stats['improved']}"
    )

    print(
        f"  Correct -> Wrong: "
        f"{stats['worsened']}"
    )

    print(
        f"  Correct -> Correct: "
        f"{stats['unchanged_correct']}"
    )

    print(
        f"  Wrong -> Wrong: "
        f"{stats['unchanged_wrong']}"
    )

    for block_idx, value in enumerate(
        stats[
            "route_disagreement"
        ].tolist()
    ):
        print(
            f"  Block {block_idx} route disagreement: "
            f"{100.0 * value:.2f}%"
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
        "\n이 비교는 official CIFAR-10 test를 사용하지 않습니다."
    )

    dataset = build_heldout_dataset(
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

    # --------------------------------------------------------
    # System A: old backbone + old refined predictor
    # --------------------------------------------------------

    old_model, old_ckpt = load_model(
        args.old_backbone,
        device,
    )

    old_predictors = load_predictors(
        args.old_predictor,
        old_model,
        device,
    )

    combinations = list(
        itertools.combinations(
            range(
                old_model.mini_heads
            ),
            old_model.direct_k,
        )
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    print(
        "\n================ SYSTEM A ================"
    )

    print(
        "Old backbone + old dynamically-refined predictor"
    )

    old_result = evaluate_system(
        name="A_OLD",
        model=old_model,
        predictors=old_predictors,
        loader=loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # System B: tuned backbone + OLD predictor
    # isolates seed-scale effect before predictor re-refinement
    # --------------------------------------------------------

    new_model, new_ckpt = load_model(
        args.new_backbone,
        device,
    )

    old_predictors_on_new = load_predictors(
        args.old_predictor,
        new_model,
        device,
    )

    print(
        "\n================ SYSTEM B ================"
    )

    print(
        "Seed-scale tuned backbone + old predictor"
    )

    seed_only_result = evaluate_system(
        name="B_SEEDSCALE_ONLY",
        model=new_model,
        predictors=old_predictors_on_new,
        loader=loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # System C: tuned backbone + predictor re-refined
    # --------------------------------------------------------

    new_predictors = load_predictors(
        args.new_predictor,
        new_model,
        device,
    )

    print(
        "\n================ SYSTEM C ================"
    )

    print(
        "Seed-scale tuned backbone + predictor refined on new dynamic states"
    )

    new_result = evaluate_system(
        name="C_SEEDSCALE_REFINED",
        model=new_model,
        predictors=new_predictors,
        loader=loader,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n================ HELD-OUT SUMMARY ================"
    )

    print_result(
        old_result
    )

    print_result(
        seed_only_result
    )

    print_result(
        new_result
    )

    # Isolate scale change.
    ab = compare_systems(
        reference=old_result,
        candidate=seed_only_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 100,
    )

    # Isolate predictor re-refinement on the tuned backbone.
    bc = compare_systems(
        reference=seed_only_result,
        candidate=new_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 200,
    )

    # Full old -> new system change.
    ac = compare_systems(
        reference=old_result,
        candidate=new_result,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 300,
    )

    print_comparison(
        "A -> B : seed-scale tuning effect",
        ab,
    )

    print_comparison(
        "B -> C : predictor re-refinement effect",
        bc,
    )

    print_comparison(
        "A -> C : full proposed update",
        ac,
    )

    print(
        "\n================ DECISION GUIDE ================"
    )

    print(
        "- A->B의 ΔCE가 음수면 seed-scale tuning이 새 held-out에서도 CE를 개선."
    )

    print(
        "- B->C의 ΔCE가 음수면 새 dynamic state에 Predictor를 다시 맞춘 효과가 실제 시스템에서도 있음."
    )

    print(
        "- A->C에서 CE와 Accuracy가 둘 다 좋아지고 95% CI도 유리하면 가장 강한 결과."
    )

    print(
        "- CE만 좋아지고 Accuracy는 불확실하면 scale 강화는 calibration/loss 측면 개선으로만 해석."
    )

    print(
        "- Accuracy CI가 0을 크게 포함하면 현재 표본에서는 정확도 우위를 확정하면 안 됨."
    )


if __name__ == "__main__":
    main()
