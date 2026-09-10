import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


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
        "--checkpoint",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "budgeted_v2_fair/mini_main/best.pt"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/checkpoints/"
            "budgeted_v2_fair/b6_b8_flip_analysis"
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=10000,
    )

    # Optional. Leave OFF first to match sparse inference.
    p.add_argument(
        "--force-dense-main",
        action="store_true",
    )

    p.add_argument(
        "--amp",
        action="store_true",
    )

    return p.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

def get_eval_transform():

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


def build_heldout_dataset(
    data_dir,
    seed,
):

    base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_eval_transform(),
    )

    if len(base) != 50000:
        raise RuntimeError(
            f"Expected CIFAR-10 train=50000, got {len(base)}"
        )

    g = torch.Generator().manual_seed(
        seed
    )

    permutation = torch.randperm(
        50000,
        generator=g,
    ).tolist()

    # Exact fair experiment split:
    #
    # train    : [0, 40000)
    # val      : [40000, 45000)
    # heldout  : [45000, 50000)

    heldout_indices = permutation[
        45000:50000
    ]

    return Subset(
        base,
        heldout_indices,
    ), heldout_indices


# ============================================================
# Model
# ============================================================

def load_checkpoint(
    path,
    device,
):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    try:
        ckpt = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            path,
            map_location=device,
        )

    if "model" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'model'."
        )

    return ckpt


def build_model_from_checkpoint(
    ckpt,
    device,
):

    config = ckpt.get(
        "config",
        {},
    )

    print("\nCheckpoint config:")
    print(config)

    # Only architecture-relevant values are taken.
    # Missing values fall back to the current V2 defaults.

    model = BudgetedMiniMainViTV2(
        img_size=int(
            config.get(
                "img_size",
                32,
            )
        ),

        patch_size=int(
            config.get(
                "patch_size",
                4,
            )
        ),

        in_chans=3,
        num_classes=10,

        embed_dim=int(
            config.get(
                "embed_dim",
                192,
            )
        ),

        depth=int(
            config.get(
                "depth",
                4,
            )
        ),

        main_heads=int(
            config.get(
                "main_heads",
                8,
            )
        ),

        mini_heads=int(
            config.get(
                "mini_heads",
                4,
            )
        ),

        mini_head_dim=int(
            config.get(
                "mini_head_dim",
                16,
            )
        ),

        direct_k=int(
            config.get(
                "direct_k",
                2,
            )
        ),

        pool_ratio=int(
            config.get(
                "pool_ratio",
                2,
            )
        ),

        mode="mini_main",

        mlp_ratio=float(
            config.get(
                "mlp_ratio",
                4.0,
            )
        ),

        drop_rate=float(
            config.get(
                "drop_rate",
                0.0,
            )
        ),

        attn_drop_rate=float(
            config.get(
                "attn_drop_rate",
                0.0,
            )
        ),

        drop_path_rate=float(
            config.get(
                "drop_path_rate",
                0.0,
            )
        ),

        route_tau=float(
            config.get(
                "route_tau",
                1.0,
            )
        ),

        bind_tau=float(
            config.get(
                "bind_tau",
                1.0,
            )
        ),
    ).to(device)

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


# ============================================================
# Metrics per sample
# ============================================================

def batch_statistics(
    logits,
    targets,
):

    probs = logits.softmax(
        dim=-1
    )

    pred = logits.argmax(
        dim=-1
    )

    correct = (
        pred == targets
    )

    # Per-sample CE
    losses = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    # Probability assigned to GT class
    true_prob = probs.gather(
        1,
        targets[:, None],
    ).squeeze(1)

    # Top-1 probability
    top2_prob, top2_idx = torch.topk(
        probs,
        k=2,
        dim=-1,
    )

    top1_conf = top2_prob[:, 0]

    # Predicted top1 - predicted top2 probability
    top1_top2_margin = (
        top2_prob[:, 0]
        -
        top2_prob[:, 1]
    )

    # True probability vs strongest wrong probability
    wrong_probs = probs.clone()

    wrong_probs.scatter_(
        1,
        targets[:, None],
        -float("inf"),
    )

    strongest_wrong_prob = (
        wrong_probs.max(
            dim=1
        ).values
    )

    true_prob_margin = (
        true_prob
        -
        strongest_wrong_prob
    )

    # Same margin in raw logits
    true_logit = logits.gather(
        1,
        targets[:, None],
    ).squeeze(1)

    wrong_logits = logits.clone()

    wrong_logits.scatter_(
        1,
        targets[:, None],
        -float("inf"),
    )

    strongest_wrong_logit = (
        wrong_logits.max(
            dim=1
        ).values
    )

    true_logit_margin = (
        true_logit
        -
        strongest_wrong_logit
    )

    # Predictive entropy
    entropy = -(
        probs
        *
        torch.log(
            probs.clamp_min(1e-12)
        )
    ).sum(dim=1)

    return {
        "logits": logits.cpu(),
        "pred": pred.cpu(),
        "correct": correct.cpu(),
        "loss": losses.cpu(),

        "true_prob": true_prob.cpu(),
        "top1_conf": top1_conf.cpu(),

        "top1_top2_margin":
            top1_top2_margin.cpu(),

        "true_prob_margin":
            true_prob_margin.cpu(),

        "true_logit_margin":
            true_logit_margin.cpu(),

        "entropy":
            entropy.cpu(),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_budget(
    model,
    loader,
    budget,
    device,
    use_amp=False,
    force_dense_main=False,
):

    storage = {
        "targets": [],
        "logits": [],
        "pred": [],
        "correct": [],
        "loss": [],

        "true_prob": [],
        "top1_conf": [],
        "top1_top2_margin": [],
        "true_prob_margin": [],
        "true_logit_margin": [],
        "entropy": [],
    }

    autocast_enabled = (
        use_amp
        and device.type == "cuda"
    )

    for batch_idx, (
        images,
        targets,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):

            logits = model(
                images,
                budget=budget,
                force_dense_main=force_dense_main,
            )

        stats = batch_statistics(
            logits.float(),
            targets,
        )

        storage[
            "targets"
        ].append(
            targets.cpu()
        )

        for key in stats:
            storage[key].append(
                stats[key]
            )

        if (
            batch_idx % 10 == 0
            or
            batch_idx + 1 == len(loader)
        ):
            print(
                f"B={budget} "
                f"[{batch_idx + 1}/{len(loader)}]"
            )

    output = {}

    for key, values in storage.items():
        output[key] = torch.cat(
            values,
            dim=0,
        )

    return output


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_mean_ci(
    values,
    repeats=10000,
    seed=42,
):

    values = torch.as_tensor(
        values,
        dtype=torch.float64,
    )

    n = values.numel()

    g = torch.Generator().manual_seed(
        seed
    )

    boot_means = torch.empty(
        repeats,
        dtype=torch.float64,
    )

    # Chunked to avoid allocating
    # [10000, 5000] all at once.

    chunk = 200

    done = 0

    while done < repeats:

        r = min(
            chunk,
            repeats - done,
        )

        idx = torch.randint(
            low=0,
            high=n,
            size=(r, n),
            generator=g,
        )

        samples = values[
            idx
        ]

        boot_means[
            done:done + r
        ] = samples.mean(
            dim=1
        )

        done += r

    q = torch.quantile(
        boot_means,
        torch.tensor(
            [0.025, 0.975],
            dtype=torch.float64,
        ),
    )

    return (
        values.mean().item(),
        q[0].item(),
        q[1].item(),
    )


# ============================================================
# Exact McNemar
# ============================================================

def exact_mcnemar(
    b6_correct_b8_wrong,
    b6_wrong_b8_correct,
):

    b = int(
        b6_correct_b8_wrong
    )

    c = int(
        b6_wrong_b8_correct
    )

    n = b + c

    if n == 0:
        return 1.0

    try:
        from scipy.stats import binomtest

        result = binomtest(
            k=min(b, c),
            n=n,
            p=0.5,
            alternative="two-sided",
        )

        return float(
            result.pvalue
        )

    except ImportError:

        # Exact two-sided binomial fallback
        k = min(b, c)

        tail = sum(
            math.comb(n, i)
            for i in range(
                0,
                k + 1,
            )
        ) / (2 ** n)

        return min(
            1.0,
            2.0 * tail,
        )


# ============================================================
# Printing helpers
# ============================================================

def mean_median(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return (
        np.mean(x),
        np.median(x),
    )


def print_metric_group(
    name,
    mask,
    b6,
    b8,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    n = int(
        mask.sum()
    )

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"N                         : {n}"
    )

    if n == 0:
        return

    keys = [
        (
            "True-class probability",
            "true_prob",
        ),
        (
            "Top-1 confidence",
            "top1_conf",
        ),
        (
            "Top1-top2 prob margin",
            "top1_top2_margin",
        ),
        (
            "True-vs-wrong prob margin",
            "true_prob_margin",
        ),
        (
            "True-vs-wrong logit margin",
            "true_logit_margin",
        ),
        (
            "Entropy",
            "entropy",
        ),
    ]

    for label, key in keys:

        x6 = b6[key][mask]
        x8 = b8[key][mask]

        m6, med6 = mean_median(
            x6
        )

        m8, med8 = mean_median(
            x8
        )

        delta = (
            np.asarray(x8)
            -
            np.asarray(x6)
        )

        dmean, dmedian = (
            mean_median(delta)
        )

        print()
        print(label)

        print(
            f"  B6 mean / median        : "
            f"{m6:.6f} / {med6:.6f}"
        )

        print(
            f"  B8 mean / median        : "
            f"{m8:.6f} / {med8:.6f}"
        )

        print(
            f"  Δ(B8-B6) mean / median : "
            f"{dmean:+.6f} / {dmedian:+.6f}"
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

    print("=" * 70)
    print("B6 vs B8 PAIRED FLIP ANALYSIS")
    print("=" * 70)

    print(
        f"Device                    : {device}"
    )

    print(
        f"Checkpoint                : {args.checkpoint}"
    )

    print(
        f"Force dense Main          : {args.force_dense_main}"
    )

    print(
        f"AMP                       : {args.amp}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ckpt = load_checkpoint(
        args.checkpoint,
        device,
    )

    model = build_model_from_checkpoint(
        ckpt,
        device,
    )

    print()
    print(
        "Model:"
        f" depth={model.depth},"
        f" Mini={model.mini_heads},"
        f" Main={model.main_heads},"
        f" direct_k={model.direct_k}"
    )

    # --------------------------------------------------------
    # Heldout
    # --------------------------------------------------------

    heldout_set, heldout_indices = (
        build_heldout_dataset(
            args.data_dir,
            args.seed,
        )
    )

    loader = DataLoader(
        heldout_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    print()
    print(
        f"Heldout samples           : {len(heldout_set)}"
    )

    # --------------------------------------------------------
    # B6/B8
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluating B=6")
    print("=" * 70)

    b6 = evaluate_budget(
        model=model,
        loader=loader,
        budget=6,
        device=device,
        use_amp=args.amp,
        force_dense_main=args.force_dense_main,
    )

    print()
    print("=" * 70)
    print("Evaluating B=8")
    print("=" * 70)

    b8 = evaluate_budget(
        model=model,
        loader=loader,
        budget=8,
        device=device,
        use_amp=args.amp,
        force_dense_main=args.force_dense_main,
    )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    y = b6[
        "targets"
    ].numpy()

    if not torch.equal(
        b6["targets"],
        b8["targets"],
    ):
        raise RuntimeError(
            "B6 and B8 target order differs."
        )

    c6 = b6[
        "correct"
    ].numpy()

    c8 = b8[
        "correct"
    ].numpy()

    p6 = b6[
        "pred"
    ].numpy()

    p8 = b8[
        "pred"
    ].numpy()

    # --------------------------------------------------------
    # Reproduce headline metrics
    # --------------------------------------------------------

    acc6 = (
        c6.mean() * 100.0
    )

    acc8 = (
        c8.mean() * 100.0
    )

    ce6 = (
        b6["loss"]
        .mean()
        .item()
    )

    ce8 = (
        b8["loss"]
        .mean()
        .item()
    )

    print()
    print("=" * 70)
    print("HEADLINE RESULT")
    print("=" * 70)

    print(
        f"B6 CE                     : {ce6:.8f}"
    )

    print(
        f"B6 Accuracy               : {acc6:.4f}%"
    )

    print()
    print(
        f"B8 CE                     : {ce8:.8f}"
    )

    print(
        f"B8 Accuracy               : {acc8:.4f}%"
    )

    print()
    print(
        f"ΔCE B8-B6                 : {ce8-ce6:+.8f}"
    )

    print(
        f"ΔAccuracy B8-B6           : {acc8-acc6:+.4f}%p"
    )

    # --------------------------------------------------------
    # Flip table
    # --------------------------------------------------------

    cc = (
        c6
        &
        c8
    )

    cw = (
        c6
        &
        ~c8
    )

    wc = (
        ~c6
        &
        c8
    )

    ww = (
        ~c6
        &
        ~c8
    )

    print()
    print("=" * 70)
    print("PAIRED PREDICTION FLIPS")
    print("=" * 70)

    print(
        f"B6 correct -> B8 correct : {cc.sum():4d}"
    )

    print(
        f"B6 correct -> B8 wrong   : {cw.sum():4d}"
    )

    print(
        f"B6 wrong   -> B8 correct : {wc.sum():4d}"
    )

    print(
        f"B6 wrong   -> B8 wrong   : {ww.sum():4d}"
    )

    print()
    print(
        f"Total prediction changes  : "
        f"{(p6 != p8).sum():4d}"
    )

    print(
        f"Same predicted class      : "
        f"{(p6 == p8).sum():4d}"
    )

    # Accuracy difference must equal WC-CW
    net_correct_change = (
        int(wc.sum())
        -
        int(cw.sum())
    )

    print()
    print(
        f"Net correct change        : "
        f"{net_correct_change:+d} samples"
    )

    print(
        f"Expected Δaccuracy        : "
        f"{100 * net_correct_change / len(y):+.4f}%p"
    )

    # --------------------------------------------------------
    # McNemar
    # --------------------------------------------------------

    mcnemar_p = exact_mcnemar(
        cw.sum(),
        wc.sum(),
    )

    print()
    print("=" * 70)
    print("EXACT McNEMAR TEST")
    print("=" * 70)

    print(
        f"B6 correct -> B8 wrong   : {cw.sum()}"
    )

    print(
        f"B6 wrong -> B8 correct   : {wc.sum()}"
    )

    print(
        f"Discordant pairs          : "
        f"{cw.sum() + wc.sum()}"
    )

    print(
        f"Exact two-sided p-value   : {mcnemar_p:.8f}"
    )

    if mcnemar_p < 0.05:
        print(
            "Result                    : "
            "significant at alpha=0.05"
        )
    else:
        print(
            "Result                    : "
            "NOT significant at alpha=0.05"
        )

    # --------------------------------------------------------
    # Paired bootstrap accuracy CI
    # --------------------------------------------------------

    acc_delta_per_sample = (
        c8.astype(
            np.float64
        )
        -
        c6.astype(
            np.float64
        )
    ) * 100.0

    (
        dacc,
        dacc_lo,
        dacc_hi,
    ) = bootstrap_mean_ci(
        acc_delta_per_sample,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    ce_delta_per_sample = (
        b8["loss"].numpy()
        -
        b6["loss"].numpy()
    )

    (
        dce,
        dce_lo,
        dce_hi,
    ) = bootstrap_mean_ci(
        ce_delta_per_sample,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1,
    )

    print()
    print("=" * 70)
    print("PAIRED BOOTSTRAP")
    print("=" * 70)

    print(
        "Accuracy Δ(B8-B6)"
    )

    print(
        f"  Mean                    : {dacc:+.4f}%p"
    )

    print(
        f"  95% CI                  : "
        f"[{dacc_lo:+.4f}, {dacc_hi:+.4f}]%p"
    )

    print()
    print(
        "CE Δ(B8-B6)"
    )

    print(
        f"  Mean                    : {dce:+.8f}"
    )

    print(
        f"  95% CI                  : "
        f"[{dce_lo:+.8f}, {dce_hi:+.8f}]"
    )

    # --------------------------------------------------------
    # Metric arrays
    # --------------------------------------------------------

    b6_np = {}
    b8_np = {}

    metric_keys = [
        "true_prob",
        "top1_conf",
        "top1_top2_margin",
        "true_prob_margin",
        "true_logit_margin",
        "entropy",
    ]

    for key in metric_keys:

        b6_np[key] = (
            b6[key]
            .numpy()
        )

        b8_np[key] = (
            b8[key]
            .numpy()
        )

    # --------------------------------------------------------
    # Confidence / margin analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONFIDENCE / MARGIN ANALYSIS")
    print("=" * 70)

    print_metric_group(
        "ALL SAMPLES",
        np.ones(
            len(y),
            dtype=bool,
        ),
        b6_np,
        b8_np,
    )

    print_metric_group(
        "B6 CORRECT -> B8 CORRECT",
        cc,
        b6_np,
        b8_np,
    )

    print_metric_group(
        "B6 CORRECT -> B8 WRONG  [DEGRADED]",
        cw,
        b6_np,
        b8_np,
    )

    print_metric_group(
        "B6 WRONG -> B8 CORRECT  [RESCUED]",
        wc,
        b6_np,
        b8_np,
    )

    print_metric_group(
        "B6 WRONG -> B8 WRONG",
        ww,
        b6_np,
        b8_np,
    )

    # --------------------------------------------------------
    # Boundary analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DECISION-BOUNDARY FLIPS")
    print("=" * 70)

    for name, mask in [
        (
            "DEGRADED B6 correct -> B8 wrong",
            cw,
        ),
        (
            "RESCUED B6 wrong -> B8 correct",
            wc,
        ),
    ]:

        n = mask.sum()

        print()
        print(name)
        print(
            f"N                         : {n}"
        )

        if n == 0:
            continue

        b6_margin = (
            b6_np[
                "true_logit_margin"
            ][mask]
        )

        b8_margin = (
            b8_np[
                "true_logit_margin"
            ][mask]
        )

        print(
            f"B6 true-logit margin "
            f"mean/median               : "
            f"{b6_margin.mean():+.6f} / "
            f"{np.median(b6_margin):+.6f}"
        )

        print(
            f"B8 true-logit margin "
            f"mean/median               : "
            f"{b8_margin.mean():+.6f} / "
            f"{np.median(b8_margin):+.6f}"
        )

        print(
            f"Mean margin shift         : "
            f"{(b8_margin-b6_margin).mean():+.6f}"
        )

        b6_true_prob = (
            b6_np[
                "true_prob"
            ][mask]
        )

        b8_true_prob = (
            b8_np[
                "true_prob"
            ][mask]
        )

        print(
            f"B6 true prob mean         : "
            f"{b6_true_prob.mean():.6f}"
        )

        print(
            f"B8 true prob mean         : "
            f"{b8_true_prob.mean():.6f}"
        )

        print(
            f"Mean true-prob shift      : "
            f"{(b8_true_prob-b6_true_prob).mean():+.6f}"
        )

    # --------------------------------------------------------
    # Save per-sample CSV
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    group = np.full(
        len(y),
        "",
        dtype=object,
    )

    group[cc] = "correct_correct"
    group[cw] = "correct_wrong"
    group[wc] = "wrong_correct"
    group[ww] = "wrong_wrong"

    df = pd.DataFrame(
        {
            "heldout_position":
                np.arange(len(y)),

            "original_cifar_train_index":
                heldout_indices,

            "target":
                y,

            "b6_pred":
                p6,

            "b8_pred":
                p8,

            "b6_correct":
                c6.astype(int),

            "b8_correct":
                c8.astype(int),

            "flip_group":
                group,

            "prediction_changed":
                (p6 != p8).astype(int),

            "b6_loss":
                b6["loss"].numpy(),

            "b8_loss":
                b8["loss"].numpy(),

            "delta_loss_b8_minus_b6":
                (
                    b8["loss"].numpy()
                    -
                    b6["loss"].numpy()
                ),

            "b6_true_prob":
                b6_np["true_prob"],

            "b8_true_prob":
                b8_np["true_prob"],

            "delta_true_prob":
                (
                    b8_np["true_prob"]
                    -
                    b6_np["true_prob"]
                ),

            "b6_top1_conf":
                b6_np["top1_conf"],

            "b8_top1_conf":
                b8_np["top1_conf"],

            "b6_top1_top2_margin":
                b6_np[
                    "top1_top2_margin"
                ],

            "b8_top1_top2_margin":
                b8_np[
                    "top1_top2_margin"
                ],

            "b6_true_prob_margin":
                b6_np[
                    "true_prob_margin"
                ],

            "b8_true_prob_margin":
                b8_np[
                    "true_prob_margin"
                ],

            "b6_true_logit_margin":
                b6_np[
                    "true_logit_margin"
                ],

            "b8_true_logit_margin":
                b8_np[
                    "true_logit_margin"
                ],

            "delta_true_logit_margin":
                (
                    b8_np[
                        "true_logit_margin"
                    ]
                    -
                    b6_np[
                        "true_logit_margin"
                    ]
                ),

            "b6_entropy":
                b6_np["entropy"],

            "b8_entropy":
                b8_np["entropy"],
        }
    )

    csv_path = (
        output_dir
        /
        "b6_b8_per_sample.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # Flip-only CSV
    flip_df = df[
        (df["flip_group"] == "correct_wrong")
        |
        (df["flip_group"] == "wrong_correct")
    ].copy()

    flip_path = (
        output_dir
        /
        "b6_b8_correctness_flips.csv"
    )

    flip_df.to_csv(
        flip_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "n": len(y),

        "b6_ce": ce6,
        "b8_ce": ce8,

        "b6_acc": acc6,
        "b8_acc": acc8,

        "delta_ce_b8_minus_b6":
            ce8 - ce6,

        "delta_acc_b8_minus_b6":
            acc8 - acc6,

        "correct_correct":
            int(cc.sum()),

        "correct_wrong":
            int(cw.sum()),

        "wrong_correct":
            int(wc.sum()),

        "wrong_wrong":
            int(ww.sum()),

        "prediction_changed":
            int(
                (p6 != p8).sum()
            ),

        "mcnemar_p":
            mcnemar_p,

        "accuracy_delta_ci95_low":
            dacc_lo,

        "accuracy_delta_ci95_high":
            dacc_hi,

        "ce_delta_ci95_low":
            dce_lo,

        "ce_delta_ci95_high":
            dce_hi,
    }

    summary_path = (
        output_dir
        /
        "b6_b8_summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        csv_path
    )

    print(
        flip_path
    )

    print(
        summary_path
    )

    # --------------------------------------------------------
    # Interpretation helper
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERPRETATION CHECKLIST")
    print("=" * 70)

    print(
        """
1. McNemar p >= 0.05 이고
   bootstrap accuracy CI가 0을 포함
   → 현재 -0.22%p drop은 통계적으로 명확하지 않음.

2. B6 correct -> B8 wrong 숫자와
   B6 wrong -> B8 correct 숫자가 비슷
   → B8이 구조적으로 악화한다기보다
     decision-boundary sample이 서로 교환되는 현상일 가능성.

3. DEGRADED group에서
   B6 true-logit margin 자체가 작으면
   → B6에서 간신히 맞던 boundary sample이
     추가 heads 때문에 뒤집힌 것.

4. 전체 CE는 좋아지면서 accuracy만 낮아지고,
   correct-correct 그룹의 true probability가 증가한다면
   → B8이 전체적인 probabilistic prediction을
     악화시킨 것은 아님.

5. DEGRADED 수가 RESCUED보다 지속적으로 많고
   McNemar도 유의하다면
   → B6->B8에서 추가되는 heads의
     marginal-head ablation을 다음으로 해야 함.
        """
    )


if __name__ == "__main__":
    main()