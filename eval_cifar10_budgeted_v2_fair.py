# -*- coding: utf-8 -*-
"""
Heldout evaluation for the fair 30-epoch experiment.

Heldout:
  CIFAR-10 official train set permutation [45000,50000), n=5000.
Official CIFAR-10 test is NOT used.

Reports:
1) MiniMain and MainOnly B-wise CE / accuracy.
2) Same-B paired CE comparisons:
      MM B2 vs MO B2
      MM B4 vs MO B4
      MM B6 vs MO B6
      MM B8 vs MO B8
3) Compute-matched-ish paired comparisons:
      MM B2 vs MO B4
      MM B4 vs MO B6
      MM B6 vs MO B8
4) MiniMain internal B0/B2/B4/B6/B8 curve.
5) Attention MAC and approximate whole-block MAC.
6) Actual sparse inference latency.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content/cifar10")
    p.add_argument("--mini-main-checkpoint", required=True)
    p.add_argument("--main-only-checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    p.add_argument("--latency-repeats", type=int, default=30)
    p.add_argument(
        "--output",
        default="./outputs/budgeted_v2_fair_heldout.json",
    )
    return p.parse_args()


def load_model(path, device):
    ck = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    c = ck["config"]
    m = BudgetedMiniMainViTV2(
        img_size=c["img_size"],
        patch_size=c["patch_size"],
        num_classes=c["num_classes"],
        embed_dim=c["embed_dim"],
        depth=c["depth"],
        main_heads=c["main_heads"],
        mini_heads=c["mini_heads"],
        mini_head_dim=c["mini_head_dim"],
        direct_k=c["direct_k"],
        pool_ratio=c["pool_ratio"],
        mode=c["mode"],
    )
    m.load_state_dict(ck["model"], strict=True)
    m.to(device).eval()
    return m, ck


def heldout_loader(args):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    base = datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=tf,
    )

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(
        50000,
        generator=g,
    ).tolist()

    ds = Subset(base, perm[45000:50000])

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect(model, loader, budget, device):
    losses = []
    correct = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(
            x,
            budget=budget,
            force_dense_main=False,
        )

        losses.append(
            F.cross_entropy(
                logits,
                y,
                reduction="none",
            ).cpu()
        )
        correct.append(
            logits.argmax(-1).eq(y).cpu()
        )

    return {
        "losses": torch.cat(losses),
        "correct": torch.cat(correct),
    }


def summary(result):
    return {
        "ce": result["losses"].mean().item(),
        "acc": (
            100.0
            * result["correct"].float().mean().item()
        ),
    }


def bootstrap_ci(delta, repeats, seed):
    d = delta.float().cpu()
    n = d.numel()

    g = torch.Generator().manual_seed(seed)

    vals = []
    done = 0
    while done < repeats:
        r = min(200, repeats - done)
        idx = torch.randint(
            0,
            n,
            (r, n),
            generator=g,
        )
        vals.append(
            d[idx].mean(dim=1)
        )
        done += r

    v = torch.cat(vals)

    return {
        "mean": d.mean().item(),
        "ci95": [
            torch.quantile(v, 0.025).item(),
            torch.quantile(v, 0.975).item(),
        ],
    }


def approx_block_macs(model, budget):
    """
    Attention MAC from model estimator
    + MLP linear MACs.
    LayerNorm / GELU / router-control overhead is not included.
    """
    attn = model.estimate_block_attention_macs(budget)

    N = 1 + model.patch_embed.num_patches
    block0 = model.blocks[0]
    D = model.embed_dim
    hidden = block0.mlp.fc1.out_features

    mlp = (
        N * D * hidden
        + N * hidden * D
    )

    return {
        "attention_macs": attn["attention_total_macs"],
        "mini_macs": attn["mini_macs"],
        "main_macs": attn["main_macs"],
        "mlp_linear_macs": float(mlp),
        "approx_block_total_macs": float(
            attn["attention_total_macs"] + mlp
        ),
        "note": (
            "Approximate block MAC excludes LayerNorm, GELU, "
            "routing/control and other small elementwise overhead."
        ),
    }


@torch.no_grad()
def latency(model, budget, device, batch_size, repeats):
    x = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
    )

    # warmup
    for _ in range(10):
        model(
            x,
            budget=budget,
            force_dense_main=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        model(
            x,
            budget=budget,
            force_dense_main=False,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(
            (time.perf_counter() - t0) * 1000.0
        )

    times = sorted(times)

    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "batch_size": batch_size,
        "repeats": repeats,
    }


def print_pair(name, result):
    lo, hi = result["ci95"]
    print(
        f"{name}: dCE={result['mean']:+.8f} "
        f"95%CI[{lo:+.8f},{hi:+.8f}]"
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("FAIR 4 MINI / 8 MAIN HELDOUT EVALUATION")
    print("=" * 100)
    print("device:", device)
    print(
        "Heldout: CIFAR-10 train permutation "
        "[45000,50000), n=5000"
    )
    print("Official CIFAR-10 test: NOT USED")

    mm, mm_ck = load_model(
        args.mini_main_checkpoint,
        device,
    )
    mo, mo_ck = load_model(
        args.main_only_checkpoint,
        device,
    )

    if mm.mode != "mini_main":
        raise ValueError(
            "mini-main checkpoint has wrong mode."
        )
    if mo.mode != "main_only":
        raise ValueError(
            "main-only checkpoint has wrong mode."
        )

    print(
        "MiniMain checkpoint epoch:",
        mm_ck.get("epoch"),
    )
    print(
        "MainOnly checkpoint epoch:",
        mo_ck.get("epoch"),
    )

    loader = heldout_loader(args)

    raw_mm = {}
    raw_mo = {}

    report = {
        "heldout": "[45000,50000)",
        "n": 5000,
        "official_test_used": False,
        "mini_main": {},
        "main_only": {},
        "paired_same_budget": {},
        "paired_compute_matched": {},
        "mini_internal": {},
        "macs": {},
        "latency": {},
    }

    print("\n[MiniMain]")
    for b in [0, 2, 4, 6, 8]:
        r = collect(mm, loader, b, device)
        raw_mm[b] = r
        s = summary(r)
        report["mini_main"][str(b)] = s
        print(
            f"B={b}: CE={s['ce']:.6f} "
            f"Acc={s['acc']:.2f}%"
        )

    print("\n[MainOnly]")
    for b in [2, 4, 6, 8]:
        r = collect(mo, loader, b, device)
        raw_mo[b] = r
        s = summary(r)
        report["main_only"][str(b)] = s
        print(
            f"B={b}: CE={s['ce']:.6f} "
            f"Acc={s['acc']:.2f}%"
        )

    print(
        "\n[Same-budget paired comparison: "
        "MiniMain - MainOnly]"
    )
    for i, b in enumerate([2, 4, 6, 8]):
        d = bootstrap_ci(
            raw_mm[b]["losses"]
            - raw_mo[b]["losses"],
            args.bootstrap_repeats,
            args.seed + 100 + i,
        )
        report["paired_same_budget"][str(b)] = d
        print_pair(f"B={b}", d)

    print(
        "\n[Compute-matched-ish paired comparison: "
        "MiniMain - MainOnly]"
    )
    compute_pairs = [
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    for i, (b_mm, b_mo) in enumerate(compute_pairs):
        d = bootstrap_ci(
            raw_mm[b_mm]["losses"]
            - raw_mo[b_mo]["losses"],
            args.bootstrap_repeats,
            args.seed + 300 + i,
        )
        key = f"MM_B{b_mm}_minus_MO_B{b_mo}"
        report["paired_compute_matched"][key] = d
        print_pair(
            f"MiniMain B={b_mm} - MainOnly B={b_mo}",
            d,
        )

    print("\n[MiniMain internal budget gaps]")
    for i, (a, b) in enumerate([
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (4, 8),
    ]):
        d = bootstrap_ci(
            raw_mm[a]["losses"]
            - raw_mm[b]["losses"],
            args.bootstrap_repeats,
            args.seed + 500 + i,
        )
        key = f"B{a}_minus_B{b}"
        report["mini_internal"][key] = d
        print_pair(key, d)

    print("\n[MAC estimates]")
    report["macs"] = {
        "mini_main": {},
        "main_only": {},
    }

    for name, model, budgets in [
        ("mini_main", mm, [0, 2, 4, 6, 8]),
        ("main_only", mo, [2, 4, 6, 8]),
    ]:
        for b in budgets:
            z = approx_block_macs(model, b)
            report["macs"][name][str(b)] = z
            print(
                f"{name:10s} B={b}: "
                f"attn={z['attention_macs']/1e6:.3f}M, "
                f"approx block="
                f"{z['approx_block_total_macs']/1e6:.3f}M"
            )

    print("\n[Sparse inference latency]")
    report["latency"] = {
        "mini_main": {},
        "main_only": {},
    }

    for name, model, budgets in [
        ("mini_main", mm, [0, 2, 4, 6, 8]),
        ("main_only", mo, [2, 4, 6, 8]),
    ]:
        for b in budgets:
            z = latency(
                model,
                b,
                device,
                args.batch_size,
                args.latency_repeats,
            )
            report["latency"][name][str(b)] = z
            print(
                f"{name:10s} B={b}: "
                f"mean={z['mean_ms']:.3f}ms "
                f"median={z['median_ms']:.3f}ms "
                f"(batch={args.batch_size})"
            )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print(
        "1) Same-B dCE < 0: Mini information helps "
        "when the number of active Main heads is matched."
    )
    print(
        "2) Compute-matched dCE < 0: strongest signal for "
        "'Mini can replace some Main computation'."
    )
    print(
        "3) Mini B=0: checks whether Mini alone forms a "
        "usable base representation."
    )
    print(
        "4) Mini B=4 vs B=8 gap: checks whether half the "
        "Main heads approach Full Main."
    )
    print(
        "5) MAC + measured latency must both be reported. "
        "Attention MAC alone is not whole-model compute."
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()
