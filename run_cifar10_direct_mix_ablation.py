import argparse
import itertools
import os
import random
from contextlib import contextmanager, nullcontext
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT


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
        "--seed",
        type=int,
        default=42,
    )

    # Reuse an already-consumed diagnostic split.
    # No new held-out data are consumed.
    p.add_argument(
        "--heldout-start",
        type=int,
        default=12596,
    )

    p.add_argument(
        "--heldout-samples",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--amp",
        action="store_true",
    )

    # Previously selected robust fixed route.
    p.add_argument(
        "--robust-b0",
        type=str,
        default="1,2",
    )

    p.add_argument(
        "--robust-b1",
        type=str,
        default="1,3",
    )

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
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


def build_model_from_config(config):
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


def load_model(path, device):
    ckpt = load_file(
        path,
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

    return model


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


def build_reused_heldout(args):
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

    start = args.heldout_start
    end = (
        start
        +
        args.heldout_samples
    )

    if start < 0 or end > len(base):
        raise ValueError(
            f"Invalid held-out range [{start}, {end}) "
            f"for CIFAR-10 train size {len(base)}."
        )

    print(
        f"\nReused diagnostic split: "
        f"[{start}, {end}) "
        f"n={end-start}"
    )

    print(
        "This split was already consumed in prior diagnostics."
    )

    print(
        "No new held-out data and no official CIFAR-10 test are used."
    )

    return Subset(
        base,
        permutation[start:end],
    )


# ============================================================
# Route helpers
# ============================================================

def build_route_space(model):
    combinations = list(
        itertools.combinations(
            range(model.mini_heads),
            model.direct_k,
        )
    )

    routes = list(
        itertools.product(
            range(len(combinations)),
            repeat=model.depth,
        )
    )

    return combinations, routes


def parse_pair(text):
    pair = tuple(
        int(x.strip())
        for x in text.split(",")
        if x.strip() != ""
    )

    if len(pair) != 2:
        raise ValueError(
            f"Expected pair like '1,2', got {text!r}"
        )

    return tuple(sorted(pair))


def route_to_string(
    route,
    combinations,
):
    return " / ".join(
        f"B{block_idx}:{combinations[combo_idx]}"
        for block_idx, combo_idx in enumerate(route)
    )


def forced_pairs_from_route(
    route,
    combo_table,
    batch_size,
    device,
):
    table = combo_table.to(device)

    return [
        table[int(combo_idx)][None, :]
        .expand(batch_size, -1)
        .clone()
        for combo_idx in route
    ]


# ============================================================
# Mix ablation
# ============================================================

@contextmanager
def direct_only_mode(model):
    """
    Preserve:
      - selected Direct Mini heads
      - Mini->Main binding
      - Direct projections/seeds
      - Main attention
      - learned seed_scale

    Remove ONLY:
      - the seed sent to Main heads marked as mixed_main_mask

    This zeros the already-projected mixed Main seed AFTER the Binder.
    Therefore projection bias cannot leak a residual mixed contribution.

    With direct_k=2 and main_heads=3:
      two Main heads keep their Direct Mini seeds;
      the remaining Main head receives zero Mini seed and uses base Q only.
    """

    originals = []

    for block_idx, block in enumerate(model.blocks):
        binder = block.attn.binder
        original_forward = binder.forward

        originals.append(
            (
                binder,
                original_forward,
            )
        )

        def override_forward(
            self_binder,
            *args,
            _original_forward=original_forward,
            _block_idx=block_idx,
            **kwargs,
        ):
            output = _original_forward(
                *args,
                **kwargs,
            )

            if (
                not isinstance(output, tuple)
                or
                len(output) != 2
            ):
                raise RuntimeError(
                    f"Block {_block_idx}: Binder must return "
                    "(main_seeds, binding_info) for Direct-only ablation."
                )

            main_seeds, binding_info = output

            if "mixed_main_mask" not in binding_info:
                raise KeyError(
                    f"Block {_block_idx}: binding_info does not contain "
                    "'mixed_main_mask'."
                )

            mixed_main_mask = binding_info[
                "mixed_main_mask"
            ].bool()

            keep_mask = (
                ~mixed_main_mask
            )[
                :,
                :,
                None,
                None,
            ].to(
                dtype=main_seeds.dtype
            )

            main_seeds = (
                main_seeds
                *
                keep_mask
            )

            return (
                main_seeds,
                binding_info,
            )

        binder.forward = MethodType(
            override_forward,
            binder,
        )

    try:
        yield
    finally:
        for binder, original_forward in originals:
            binder.forward = original_forward


def amp_context(device, enabled):
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


@torch.no_grad()
def forward_forced(
    model,
    images,
    forced_pairs,
    return_info=False,
):
    return model(
        images,
        return_info=return_info,
        collect_taylor=False,
        forced_direct_indices_per_block=forced_pairs,
        forced_uniform_mix=True,
    )


# ============================================================
# Sanity check
# ============================================================

@torch.no_grad()
def run_direct_only_sanity(
    model,
    dataset,
    route,
    combo_table,
    device,
    use_amp,
):
    loader = DataLoader(
        dataset,
        batch_size=min(
            8,
            len(dataset),
        ),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    images, _ = next(
        iter(loader)
    )

    images = images.to(device)

    forced = forced_pairs_from_route(
        route,
        combo_table,
        images.shape[0],
        device,
    )

    with amp_context(
        device,
        use_amp,
    ):
        (
            logits_mix,
            info_mix,
        ) = forward_forced(
            model,
            images,
            forced,
            return_info=True,
        )

    with direct_only_mode(model):
        with amp_context(
            device,
            use_amp,
        ):
            (
                logits_direct,
                info_direct,
            ) = forward_forced(
                model,
                images,
                forced,
                return_info=True,
            )

    if len(info_mix) != len(info_direct):
        raise RuntimeError(
            "Block-info length changed during ablation."
        )

    downstream_direct_diffs = []

    for block_idx, (
        mix_info,
        direct_info,
    ) in enumerate(
        zip(
            info_mix,
            info_direct,
        )
    ):
        mix_mask = direct_info[
            "mixed_main_mask"
        ].bool()

        direct_seed = direct_info[
            "main_seeds"
        ]

        # In every ablated block, the mixed Main seed itself must
        # be exactly zero after the Binder override.
        if mix_mask.any():
            ablated_values = direct_seed[
                mix_mask
            ]

            max_abs = ablated_values.abs().max().item()

            if max_abs > 1e-7:
                raise RuntimeError(
                    f"Block {block_idx}: Direct-only mixed seed "
                    f"is not zero. max_abs={max_abs}"
                )

        direct_mask = (
            ~mix_mask
        )

        if direct_mask.any():
            original_values = mix_info[
                "main_seeds"
            ][
                direct_mask
            ]

            ablated_values = direct_info[
                "main_seeds"
            ][
                direct_mask
            ]

            max_diff = (
                original_values
                -
                ablated_values
            ).abs().max().item()

            if block_idx == 0:
                # Block 0 receives exactly the same input in both runs.
                # Therefore its Direct seeds must be unchanged.
                if max_diff > 1e-5:
                    raise RuntimeError(
                        f"Block 0: non-mixed Direct seed changed "
                        f"during the local ablation. max_diff={max_diff}"
                    )
            else:
                # From Block 1 onward, zeroing Block-0 Mix changes the
                # hidden state entering the downstream block. Direct
                # Mini contexts / Binder assignments can therefore
                # legitimately change. This is a causal downstream
                # consequence of the Mix ablation, not an ablation bug.
                downstream_direct_diffs.append(
                    (
                        block_idx,
                        max_diff,
                    )
                )

    logit_delta = (
        logits_mix.float()
        -
        logits_direct.float()
    ).abs().mean().item()

    print(
        "\n================ ABLATION SANITY ================"
    )

    print(
        "PASS: Mixed Main seeds are exactly zeroed in Direct-only mode."
    )

    print(
        "PASS: Block-0 Direct Main seeds are unchanged."
    )

    if downstream_direct_diffs:
        print(
            "NOTE: downstream Direct seeds may change because removing "
            "an earlier Mix seed changes the hidden state entering later blocks."
        )

        for block_idx, max_diff in downstream_direct_diffs:
            print(
                f"  Block {block_idx} downstream Direct-seed max diff: "
                f"{max_diff:.8f}"
            )

    print(
        f"Mean absolute logit difference on sanity batch: "
        f"{logit_delta:.8f}"
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_all_routes(
    model,
    dataset,
    routes,
    combo_table,
    device,
    batch_size,
    num_workers,
    use_amp,
    mode,
):
    if mode not in {
        "direct_mix",
        "direct_only",
    }:
        raise ValueError(mode)

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

    losses_all = []
    correct_all = []

    seen = 0

    ctx = (
        direct_only_mode(model)
        if mode == "direct_only"
        else nullcontext()
    )

    print(
        f"\nEvaluating all {len(routes)} routes: {mode}"
    )

    with ctx:
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

            batch_losses = []
            batch_correct = []

            for route in routes:
                forced = forced_pairs_from_route(
                    route,
                    combo_table,
                    B,
                    device,
                )

                with amp_context(
                    device,
                    use_amp,
                ):
                    logits = forward_forced(
                        model,
                        images,
                        forced,
                        return_info=False,
                    )

                batch_losses.append(
                    F.cross_entropy(
                        logits.float(),
                        labels,
                        reduction="none",
                    ).cpu()
                )

                batch_correct.append(
                    (
                        logits.argmax(dim=-1)
                        ==
                        labels
                    ).cpu()
                )

            losses_all.append(
                torch.stack(
                    batch_losses,
                    dim=1,
                )
            )

            correct_all.append(
                torch.stack(
                    batch_correct,
                    dim=1,
                )
            )

            seen += B

            print(
                f"{mode}: "
                f"{seen}/{len(dataset)}"
            )

    return {
        "losses":
            torch.cat(
                losses_all,
                dim=0,
            ),

        "correct":
            torch.cat(
                correct_all,
                dim=0,
            ),
    }


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(
    values,
    repeats,
    seed,
):
    values = values.float().cpu()

    n = values.numel()

    g = torch.Generator().manual_seed(
        seed
    )

    means = torch.empty(
        repeats,
        dtype=torch.float32,
    )

    done = 0
    chunk = 250

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


def paired_stats(
    reference_losses,
    candidate_losses,
    reference_correct,
    candidate_correct,
    repeats,
    seed,
):
    loss_delta = (
        candidate_losses
        -
        reference_losses
    )

    acc_delta = (
        candidate_correct.float()
        -
        reference_correct.float()
    ) * 100.0

    ce_ci = bootstrap_mean_ci(
        loss_delta,
        repeats,
        seed,
    )

    acc_ci = bootstrap_mean_ci(
        acc_delta,
        repeats,
        seed + 1,
    )

    wrong_to_correct = (
        (~reference_correct.bool())
        &
        candidate_correct.bool()
    ).sum().item()

    correct_to_wrong = (
        reference_correct.bool()
        &
        (~candidate_correct.bool())
    ).sum().item()

    return {
        "delta_ce":
            loss_delta.mean().item(),

        "ce_ci":
            ce_ci,

        "delta_acc":
            acc_delta.mean().item(),

        "acc_ci":
            acc_ci,

        "wrong_to_correct":
            wrong_to_correct,

        "correct_to_wrong":
            correct_to_wrong,
    }


def print_pair_comparison(
    title,
    stats,
):
    ce_lo, ce_hi = stats[
        "ce_ci"
    ]

    acc_lo, acc_hi = stats[
        "acc_ci"
    ]

    print(
        f"\n{title}"
    )

    print(
        f"  ΔCE(candidate-reference): "
        f"{stats['delta_ce']:+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{ce_lo:+.8f}, {ce_hi:+.8f}]"
    )

    print(
        f"  ΔAccuracy: "
        f"{stats['delta_acc']:+.3f}%p"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{acc_lo:+.3f}, {acc_hi:+.3f}]%p"
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
        "\nGoal:"
    )

    print(
        "Ablate ONLY the Remaining-Mini Mix contribution."
    )

    print(
        "Direct+Mix: two selected Mini heads are Direct-bound; "
        "remaining Mini heads are mixed and seed the unbound Main head."
    )

    print(
        "Direct-only: the same forced Direct Mini-head pair is used, "
        "and the mixed Main seed is forced to zero."
    )

    print(
        "At Block 0, Direct seeds/binding are directly controlled. "
        "At later blocks, upstream Mix removal can causally change the hidden state "
        "and therefore the downstream Binder assignment."
    )

    print(
        "\nThis is an inference-time diagnostic on an already-used split."
    )

    model = load_model(
        args.backbone_checkpoint,
        device,
    )

    if (
        model.depth != 2
        or
        model.mini_heads != 4
        or
        model.direct_k != 2
        or
        model.main_heads != 3
    ):
        print(
            "\nWarning: script was designed around "
            "depth=2, Mini=4, DirectK=2, Main=3."
        )

    dataset = build_reused_heldout(
        args
    )

    combinations, routes = (
        build_route_space(
            model
        )
    )

    combo_table = torch.tensor(
        combinations,
        dtype=torch.long,
    )

    robust_b0 = parse_pair(
        args.robust_b0
    )

    robust_b1 = parse_pair(
        args.robust_b1
    )

    if robust_b0 not in combinations:
        raise ValueError(
            f"robust B0 pair {robust_b0} not in {combinations}"
        )

    if robust_b1 not in combinations:
        raise ValueError(
            f"robust B1 pair {robust_b1} not in {combinations}"
        )

    robust_route = (
        combinations.index(
            robust_b0
        ),
        combinations.index(
            robust_b1
        ),
    )

    robust_route_idx = routes.index(
        robust_route
    )

    print(
        "\nDirect pairs:"
    )

    for idx, pair in enumerate(
        combinations
    ):
        print(
            f"  {idx}: {pair}"
        )

    print(
        f"\nControlled robust route: "
        f"{route_to_string(robust_route, combinations)}"
    )

    run_direct_only_sanity(
        model=model,
        dataset=dataset,
        route=robust_route,
        combo_table=combo_table,
        device=device,
        use_amp=use_amp,
    )

    direct_mix = evaluate_all_routes(
        model=model,
        dataset=dataset,
        routes=routes,
        combo_table=combo_table,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        mode="direct_mix",
    )

    direct_only = evaluate_all_routes(
        model=model,
        dataset=dataset,
        routes=routes,
        combo_table=combo_table,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        mode="direct_only",
    )

    mix_mean_by_route = (
        direct_mix[
            "losses"
        ].mean(
            dim=0
        )
    )

    direct_mean_by_route = (
        direct_only[
            "losses"
        ].mean(
            dim=0
        )
    )

    route_delta = (
        direct_mean_by_route
        -
        mix_mean_by_route
    )

    # Positive:
    # Direct-only has higher CE -> Mix helps.
    mix_help_count = (
        route_delta
        >
        0
    ).sum().item()

    direct_help_count = (
        route_delta
        <
        0
    ).sum().item()

    tie_count = len(routes) - (
        mix_help_count
        +
        direct_help_count
    )

    mix_best_idx = int(
        mix_mean_by_route.argmin().item()
    )

    direct_best_idx = int(
        direct_mean_by_route.argmin().item()
    )

    # --------------------------------------------------------
    # Controlled robust route
    # --------------------------------------------------------

    mix_robust_losses = (
        direct_mix[
            "losses"
        ][
            :,
            robust_route_idx,
        ]
    )

    direct_robust_losses = (
        direct_only[
            "losses"
        ][
            :,
            robust_route_idx,
        ]
    )

    mix_robust_correct = (
        direct_mix[
            "correct"
        ][
            :,
            robust_route_idx,
        ]
    )

    direct_robust_correct = (
        direct_only[
            "correct"
        ][
            :,
            robust_route_idx,
        ]
    )

    # reference = Direct-only, candidate = Direct+Mix
    robust_stats = paired_stats(
        reference_losses=direct_robust_losses,
        candidate_losses=mix_robust_losses,
        reference_correct=direct_robust_correct,
        candidate_correct=mix_robust_correct,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 100,
    )

    # --------------------------------------------------------
    # Route-averaged sample metric
    #
    # For each image, average CE over the same 36 routes,
    # then compare Mix vs Direct-only.
    # This asks whether Mix helps broadly rather than only on
    # one chosen route.
    # --------------------------------------------------------

    sample_mix_route_mean = (
        direct_mix[
            "losses"
        ].mean(
            dim=1
        )
    )

    sample_direct_route_mean = (
        direct_only[
            "losses"
        ].mean(
            dim=1
        )
    )

    route_avg_delta = (
        sample_mix_route_mean
        -
        sample_direct_route_mean
    )

    route_avg_ci = bootstrap_mean_ci(
        route_avg_delta,
        args.bootstrap_repeats,
        args.seed + 200,
    )

    # --------------------------------------------------------
    # Oracle in each architecture.
    # This is diagnostic, not a controlled same-route comparison.
    # --------------------------------------------------------

    mix_oracle_losses, mix_oracle_idx = (
        direct_mix[
            "losses"
        ].min(
            dim=1
        )
    )

    direct_oracle_losses, direct_oracle_idx = (
        direct_only[
            "losses"
        ].min(
            dim=1
        )
    )

    oracle_delta = (
        mix_oracle_losses
        -
        direct_oracle_losses
    )

    oracle_ci = bootstrap_mean_ci(
        oracle_delta,
        args.bootstrap_repeats,
        args.seed + 300,
    )

    print(
        "\n================ DIRECT vs MIX ABLATION SUMMARY ================"
    )

    print(
        "\nControlled robust fixed route:"
    )

    print(
        f"  {route_to_string(robust_route, combinations)}"
    )

    print(
        f"  Direct-only CE: "
        f"{direct_robust_losses.mean().item():.6f}"
    )

    print(
        f"  Direct-only Accuracy: "
        f"{100.0 * direct_robust_correct.float().mean().item():.2f}%"
    )

    print(
        f"  Direct+Mix CE: "
        f"{mix_robust_losses.mean().item():.6f}"
    )

    print(
        f"  Direct+Mix Accuracy: "
        f"{100.0 * mix_robust_correct.float().mean().item():.2f}%"
    )

    print_pair_comparison(
        "DIRECT-ONLY -> DIRECT+MIX (same route)",
        robust_stats,
    )

    print(
        "\n================ ROUTE-WIDE EFFECT ================"
    )

    print(
        f"Routes where Mix lowers mean CE: "
        f"{mix_help_count}/{len(routes)}"
    )

    print(
        f"Routes where Direct-only lowers mean CE: "
        f"{direct_help_count}/{len(routes)}"
    )

    print(
        f"Exact mean-CE ties: "
        f"{tie_count}/{len(routes)}"
    )

    print(
        f"Mean over route-level "
        f"(Direct-only CE - Direct+Mix CE): "
        f"{route_delta.mean().item():+.8f}"
    )

    print(
        "\nPer-sample CE averaged across the SAME 36 routes:"
    )

    print(
        f"  ΔCE(Direct+Mix - Direct-only): "
        f"{route_avg_delta.mean().item():+.8f}"
    )

    print(
        f"  paired bootstrap 95% CI: "
        f"[{route_avg_ci[0]:+.8f}, {route_avg_ci[1]:+.8f}]"
    )

    print(
        "\nBest fixed route within each mode "
        "(diagnostic; selected on this reused split):"
    )

    print(
        f"  Direct+Mix best: "
        f"{route_to_string(routes[mix_best_idx], combinations)} "
        f"CE={mix_mean_by_route[mix_best_idx].item():.6f}"
    )

    print(
        f"  Direct-only best: "
        f"{route_to_string(routes[direct_best_idx], combinations)} "
        f"CE={direct_mean_by_route[direct_best_idx].item():.6f}"
    )

    print(
        "\n================ ORACLE DIAGNOSTIC ================"
    )

    print(
        f"Direct+Mix Oracle CE: "
        f"{mix_oracle_losses.mean().item():.6f}"
    )

    print(
        f"Direct-only Oracle CE: "
        f"{direct_oracle_losses.mean().item():.6f}"
    )

    print(
        f"ΔCE(Direct+Mix Oracle - Direct-only Oracle): "
        f"{oracle_delta.mean().item():+.8f}"
    )

    print(
        f"paired bootstrap 95% CI: "
        f"[{oracle_ci[0]:+.8f}, {oracle_ci[1]:+.8f}]"
    )

    same_oracle_route = (
        mix_oracle_idx
        ==
        direct_oracle_idx
    ).float().mean().item()

    print(
        f"Same per-sample Oracle whole-route: "
        f"{100.0 * same_oracle_route:.2f}%"
    )

    print(
        "\n================ DIRECT+MIX VERDICT ================"
    )

    controlled_delta = robust_stats[
        "delta_ce"
    ]

    controlled_hi = robust_stats[
        "ce_ci"
    ][1]

    broad_delta = route_avg_delta.mean().item()
    broad_hi = route_avg_ci[1]

    if (
        controlled_delta < 0
        and
        controlled_hi < 0
        and
        broad_delta < 0
        and
        broad_hi < 0
    ):
        print(
            "PASS: Remaining-Mini Mix gives a clear CE benefit."
        )

        print(
            "It improves the controlled robust route and also helps "
            "when averaging over the same 36 Direct routes."
        )

        print(
            "This directly supports keeping the 'selected Mini heads Direct, "
            "remaining Mini heads mixed instead of discarded' design."
        )

    elif (
        broad_delta < 0
        and
        broad_hi < 0
    ):
        print(
            "PARTIAL SUPPORT: Mix helps broadly across routes, "
            "but the controlled robust-route comparison is not clearly separated."
        )

    elif (
        controlled_delta < 0
        and
        controlled_hi < 0
    ):
        print(
            "ROUTE-SPECIFIC SUPPORT: Mix clearly helps the robust route, "
            "but the benefit is not broad across all Direct routes."
        )

    else:
        print(
            "NOT SUPPORTED: On this checkpoint, discarding the Remaining Mini heads "
            "is not clearly worse than mixing them."
        )

        print(
            "Do not claim a Direct+Mix advantage yet. "
            "The next step would be separately trained Direct-only vs Direct+Mix models."
        )


if __name__ == "__main__":
    main()
