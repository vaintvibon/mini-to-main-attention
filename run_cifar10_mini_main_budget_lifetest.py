# run_cifar10_mini_main_budget_lifetest.py
"""
1차 '생사 판정' 진단.

질문:
    현재 Multi-Mini -> Main Q-seeding 정보가,
    Main head 수를 제한했을 때 성능 손실을 실제로 줄여 주는가?

비교:
    같은 active Main-head 구성에서
      Seed OFF: Mini -> Main seed = 0
      Seed ON : 현재 checkpoint의 Mini -> Main seed 사용

Main heads=3, depth=2 기준:
    B=1: 3^2 = 9개 fixed Main 구성
    B=2: 3^2 = 9개 fixed Main 구성
    B=3: 1개 구성

주의:
- inactive Main head를 '계산 후 mask'한다.
  따라서 이 코드는 실제 FLOPs/latency 절감 증명이 아니다.
- 현재 checkpoint를 재학습하지 않는 inference-time 진단이다.
- 공식 CIFAR-10 test는 사용하지 않는다.
- 기본 데이터는 이미 개발 과정에서 사용한
  CIFAR-10 train permutation [12596,13596) 1000장이다.
"""

import argparse
import itertools
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.dynamic_mini_main_vit import DynamicMiniMainViT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage1_cifar10_seedscale_tuned.pt",
    )
    p.add_argument("--data-dir", type=str, default="/content/cifar10")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # 이미 개발/진단에 사용한 구간. 새 fresh split을 소모하지 않는다.
    p.add_argument("--split-start", type=int, default=12596)
    p.add_argument("--split-size", type=int, default=1000)

    # 기존 robust Mini Direct route
    p.add_argument("--block0-mini-pair", type=str, default="1,2")
    p.add_argument("--block1-mini-pair", type=str, default="1,3")

    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    p.add_argument(
        "--output",
        type=str,
        default="./outputs/mini_main_budget_lifetest.json",
    )
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfgv(cfg, key, default):
    v = cfg.get(key, default)
    return default if v is None else v


def parse_pair(text):
    xs = tuple(sorted(int(x.strip()) for x in text.split(",")))
    if len(xs) != 2 or xs[0] == xs[1]:
        raise ValueError(f"pair must be two different heads, got {text}")
    return xs


def load_model(args, device):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{args.checkpoint}"
        )

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state = None
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        raise KeyError(
            "Checkpoint has no model/state_dict/model_state_dict."
        )

    cfg = ckpt.get("config", ckpt.get("args", {}))

    model = DynamicMiniMainViT(
        img_size=cfgv(cfg, "img_size", 32),
        patch_size=cfgv(cfg, "patch_size", 4),
        in_chans=3,
        num_classes=cfgv(cfg, "num_classes", 10),
        embed_dim=cfgv(cfg, "embed_dim", 192),
        depth=cfgv(cfg, "depth", 2),
        main_heads=cfgv(cfg, "main_heads", 3),
        mini_heads=cfgv(cfg, "mini_heads", 4),
        mini_head_dim=cfgv(cfg, "mini_head_dim", 16),
        pool_ratio=cfgv(cfg, "pool_ratio", 2),
        utility_hidden_dim=cfgv(cfg, "utility_hidden_dim", 64),
        direct_k=cfgv(cfg, "direct_k", 2),
        mix_temperature=cfgv(cfg, "mix_temperature", 1.0),
        bind_dim=cfgv(cfg, "bind_dim", 64),
        bind_temperature=cfgv(cfg, "bind_temperature", 1.0),
        mlp_ratio=cfgv(cfg, "mlp_ratio", 4.0),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ).to(device)

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )
    if missing:
        raise RuntimeError(
            "Missing checkpoint keys:\n" + "\n".join(missing)
        )
    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n" + "\n".join(unexpected)
        )

    model.eval()
    return model


def build_loader(args, img_size):
    tf = []
    if img_size != 32:
        tf.append(transforms.Resize((img_size, img_size)))
    tf += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]

    base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(tf),
    )

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(base), generator=g).tolist()

    end = args.split_start + args.split_size
    if args.split_start < 0 or end > len(base):
        raise ValueError(
            f"Invalid split [{args.split_start},{end})"
        )

    ds = Subset(base, perm[args.split_start:end])

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@contextmanager
def fixed_mini_route(model, pairs):
    """
    Selector 오차를 제거한다.
    Direct Mini pair를 block별로 고정하고 Remaining Mini는 uniform mix.
    """
    if len(pairs) != len(model.blocks):
        raise ValueError("Mini route depth mismatch.")

    saved = []

    for block_idx, block in enumerate(model.blocks):
        selector = block.attn.selector
        mixer = block.attn.mixer
        pair = tuple(pairs[block_idx])

        saved.append(
            (selector, selector.forward, mixer, mixer.forward)
        )

        def selector_forward(
            self_selector,
            utility_logits,
            return_info=False,
            *,
            _pair=pair,
            _block_idx=block_idx,
        ):
            B, H = utility_logits.shape
            for h in _pair:
                if h < 0 or h >= H:
                    raise ValueError(
                        f"Block {_block_idx}: invalid Mini head H{h}"
                    )

            direct_indices = (
                torch.tensor(
                    _pair,
                    dtype=torch.long,
                    device=utility_logits.device,
                )
                .unsqueeze(0)
                .expand(B, -1)
                .clone()
            )

            direct_mask = torch.zeros(
                B, H,
                dtype=torch.bool,
                device=utility_logits.device,
            )
            direct_mask.scatter_(1, direct_indices, True)
            remaining_mask = ~direct_mask

            info = {
                "direct_indices": direct_indices,
                "ranking": utility_logits.argsort(
                    dim=-1, descending=True
                ),
            }

            if return_info:
                return direct_mask, remaining_mask, info
            return direct_mask, remaining_mask

        def mixer_forward(
            self_mixer,
            mini_contexts,
            utility_logits,
            remaining_mask,
            return_info=False,
        ):
            w = remaining_mask.to(mini_contexts.dtype)
            w = w / w.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)

            mixed = (
                mini_contexts * w[:, :, None, None]
            ).sum(dim=1)

            if return_info:
                return mixed, {"mix_weights": w}
            return mixed

        selector.forward = MethodType(
            selector_forward, selector
        )
        mixer.forward = MethodType(
            mixer_forward, mixer
        )

    try:
        yield
    finally:
        for selector, sf, mixer, mf in saved:
            selector.forward = sf
            mixer.forward = mf


class BudgetController:
    def __init__(self, depth, main_heads):
        self.depth = depth
        self.main_heads = main_heads
        self.active = [
            tuple(range(main_heads))
            for _ in range(depth)
        ]
        self.seed_on = True
        self.reconstruction_checked = [False] * depth

    def set(self, config, seed_on):
        if len(config) != self.depth:
            raise ValueError("Main config depth mismatch.")
        self.active = [
            tuple(sorted(int(h) for h in heads))
            for heads in config
        ]
        self.seed_on = bool(seed_on)


def find_proj(module):
    for name in ("proj", "out_proj", "main_proj"):
        layer = getattr(module, name, None)
        if isinstance(layer, nn.Module):
            return layer, name
    raise AttributeError(
        "Could not find Main output projection. "
        "Expected proj/out_proj/main_proj."
    )


def find_proj_drop(module):
    for name in ("proj_drop", "out_drop", "main_proj_drop"):
        layer = getattr(module, name, None)
        if isinstance(layer, nn.Module):
            return layer, name
    return nn.Identity(), "Identity"


@contextmanager
def semantic_main_budget(model, ctl):
    """
    원래 Main Attention을 그대로 실행한 뒤 head_out만 mask한다.
    실제 계산량 절감이 아니라 'head가 없다고 가정한 성능' 진단이다.
    """
    saved = []

    for block_idx, block in enumerate(model.blocks):
        main = block.attn.main_attention
        original = main.forward
        saved.append((main, original))

        def patched(
            self_main,
            x,
            main_seeds,
            return_info=False,
            *args,
            _orig=original,
            _block_idx=block_idx,
            **kwargs,
        ):
            seeds = (
                main_seeds
                if ctl.seed_on
                else torch.zeros_like(main_seeds)
            )

            original_out, info = _orig(
                x,
                seeds,
                *args,
                return_info=True,
                **kwargs,
            )

            if "head_out" not in info:
                raise KeyError(
                    "Main info lacks 'head_out'."
                )

            head_out = info["head_out"]
            B, H, N, Dh = head_out.shape

            active = ctl.active[_block_idx]
            mask = torch.zeros(
                H,
                dtype=head_out.dtype,
                device=head_out.device,
            )
            if active:
                idx = torch.tensor(
                    active,
                    dtype=torch.long,
                    device=head_out.device,
                )
                mask[idx] = 1.0

            masked = (
                head_out * mask[None, :, None, None]
            )

            if len(active) == 0:
                new_out = torch.zeros_like(original_out)
                proj_name = "bypassed"
                drop_name = "bypassed"
            else:
                merged = (
                    masked
                    .transpose(1, 2)
                    .reshape(B, N, H * Dh)
                )
                proj, proj_name = find_proj(self_main)
                drop, drop_name = find_proj_drop(self_main)
                new_out = drop(proj(merged))

            # B=all에서 우리의 reconstruction이 원래 출력과 같아야
            # 이후 masking 결과를 믿을 수 있다.
            if (
                len(active) == H
                and not ctl.reconstruction_checked[_block_idx]
            ):
                max_diff = (
                    new_out.detach() - original_out.detach()
                ).abs().max().item()

                print(
                    f"[sanity] Block {_block_idx} "
                    f"all-head reconstruction max_diff="
                    f"{max_diff:.8e} "
                    f"(proj={proj_name}, drop={drop_name})"
                )

                if max_diff > 1e-5:
                    raise RuntimeError(
                        "All-head reconstruction failed. "
                        f"Block={_block_idx}, max_diff={max_diff:.8e}. "
                        "Do not trust this ablation until the Main projection "
                        "mapping is updated for your current code."
                    )

                ctl.reconstruction_checked[_block_idx] = True

            info = dict(info)
            info["budget_head_out"] = masked
            info["active_main_mask"] = mask.bool()

            if return_info:
                return new_out, info
            return new_out

        main.forward = MethodType(patched, main)

    try:
        yield
    finally:
        for main, original in saved:
            main.forward = original


def config_key(config):
    return " | ".join(
        f"Block{i}={tuple(h)}"
        for i, h in enumerate(config)
    )


@torch.no_grad()
def eval_config(
    model,
    loader,
    device,
    ctl,
    config,
    seed_on,
):
    ctl.set(config, seed_on)
    model.eval()

    losses = []
    correct = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        losses.append(
            F.cross_entropy(
                logits, y, reduction="none"
            ).cpu()
        )
        correct.append(
            logits.argmax(dim=-1).eq(y).cpu()
        )

    return {
        "losses": torch.cat(losses),
        "correct": torch.cat(correct),
    }


def summary(r):
    return {
        "ce": r["losses"].mean().item(),
        "acc": 100.0 * r["correct"].float().mean().item(),
    }


def bootstrap_ci(delta, repeats, seed):
    """
    delta = A loss - B loss.
    음수면 A가 더 좋다.
    """
    d = delta.detach().float().cpu()
    n = d.numel()
    g = torch.Generator().manual_seed(seed)

    chunks = []
    done = 0
    while done < repeats:
        r = min(250, repeats - done)
        idx = torch.randint(
            0, n, (r, n), generator=g
        )
        chunks.append(d[idx].mean(dim=1))
        done += r

    vals = torch.cat(chunks)
    return {
        "mean": d.mean().item(),
        "ci95": [
            torch.quantile(vals, 0.025).item(),
            torch.quantile(vals, 0.975).item(),
        ],
    }


def stack_results(results, keys):
    losses = torch.stack(
        [results[k]["losses"] for k in keys]
    )
    correct = torch.stack(
        [results[k]["correct"] for k in keys]
    )
    return losses, correct


def oracle(losses, correct):
    # [num_config, num_sample]
    best = losses.argmin(dim=0)
    sample = torch.arange(losses.shape[1])
    return {
        "losses": losses[best, sample],
        "correct": correct[best, sample],
    }


def best_fixed(losses, correct, keys):
    idx = int(losses.mean(dim=1).argmin())
    return keys[idx], {
        "losses": losses[idx],
        "correct": correct[idx],
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("MINI -> MAIN BUDGET LIFE TEST")
    print("=" * 100)
    print("Device:", device)
    print("Official CIFAR-10 test: NOT USED")

    model = load_model(args, device)

    if model.depth != 2:
        raise RuntimeError(
            "This first diagnostic assumes depth=2."
        )

    print(
        f"Model: depth={model.depth}, "
        f"main_heads={model.main_heads}, "
        f"mini_heads={model.mini_heads}, "
        f"direct_k={model.direct_k}"
    )

    pair0 = parse_pair(args.block0_mini_pair)
    pair1 = parse_pair(args.block1_mini_pair)
    pairs = [pair0, pair1]

    print("Fixed Mini Direct route:")
    print("  Block0:", pair0)
    print("  Block1:", pair1)
    print("  Remaining Mini: uniform mix")

    img_size = getattr(model.patch_embed, "img_size", 32)
    loader = build_loader(args, img_size)

    print(
        f"Data: CIFAR10 train permutation "
        f"[{args.split_start},"
        f"{args.split_start + args.split_size}) "
        f"n={len(loader.dataset)}"
    )

    ctl = BudgetController(
        model.depth, model.main_heads
    )

    report = {
        "checkpoint": args.checkpoint,
        "data": {
            "seed": args.seed,
            "start": args.split_start,
            "size": args.split_size,
        },
        "mini_route": {
            "block0": list(pair0),
            "block1": list(pair1),
            "remaining_mix": "uniform",
        },
        "budgets": {},
        "cross_budget": {},
    }

    internal = {}

    with fixed_mini_route(model, pairs):
        with semantic_main_budget(model, ctl):

            # projection reconstruction sanity first
            all_heads = tuple(
                range(model.main_heads)
            )
            all_cfg = [all_heads] * model.depth

            print()
            print("[1] Reconstruction sanity...")
            _ = eval_config(
                model, loader, device, ctl,
                all_cfg, True
            )

            if not all(ctl.reconstruction_checked):
                raise RuntimeError(
                    "Not all blocks passed reconstruction sanity."
                )

            # B=1..H
            for budget in range(
                1, model.main_heads + 1
            ):
                subsets = list(
                    itertools.combinations(
                        range(model.main_heads),
                        budget,
                    )
                )
                configs = list(
                    itertools.product(
                        subsets,
                        repeat=model.depth,
                    )
                )
                keys = [
                    config_key(c) for c in configs
                ]

                print()
                print("#" * 100)
                print(
                    f"MAIN BUDGET B={budget} "
                    f"| {len(configs)} configs"
                )
                print("#" * 100)

                off = {}
                on = {}

                for i, (cfg, key) in enumerate(
                    zip(configs, keys), 1
                ):
                    roff = eval_config(
                        model, loader, device,
                        ctl, cfg, False
                    )
                    ron = eval_config(
                        model, loader, device,
                        ctl, cfg, True
                    )
                    off[key] = roff
                    on[key] = ron

                    so = summary(roff)
                    sn = summary(ron)

                    print(
                        f"[{i:02d}/{len(configs):02d}] {key} | "
                        f"OFF CE={so['ce']:.6f} "
                        f"Acc={so['acc']:.2f}% | "
                        f"ON CE={sn['ce']:.6f} "
                        f"Acc={sn['acc']:.2f}% | "
                        f"ON-OFF={sn['ce']-so['ce']:+.6f}"
                    )

                off_L, off_C = stack_results(
                    off, keys
                )
                on_L, on_C = stack_results(
                    on, keys
                )

                # 동일 구성들을 평균했을 때 seed의 순수 평균 효과
                avg_seed = bootstrap_ci(
                    on_L.mean(dim=0)
                    - off_L.mean(dim=0),
                    args.bootstrap_repeats,
                    args.seed + 1000 + budget,
                )

                off_or = oracle(off_L, off_C)
                on_or = oracle(on_L, on_C)
                oracle_seed = bootstrap_ci(
                    on_or["losses"]
                    - off_or["losses"],
                    args.bootstrap_repeats,
                    args.seed + 2000 + budget,
                )

                off_key, off_best = best_fixed(
                    off_L, off_C, keys
                )
                on_key, on_best = best_fixed(
                    on_L, on_C, keys
                )

                s_off_or = summary(off_or)
                s_on_or = summary(on_or)
                s_off_best = summary(off_best)
                s_on_best = summary(on_best)

                print()
                print(f"[B={budget} SUMMARY]")
                print(
                    "Config-average Seed effect "
                    f"(ON-OFF CE): {avg_seed['mean']:+.8f} "
                    f"95%CI[{avg_seed['ci95'][0]:+.8f},"
                    f"{avg_seed['ci95'][1]:+.8f}]"
                )
                print(
                    f"Oracle OFF: CE={s_off_or['ce']:.6f} "
                    f"Acc={s_off_or['acc']:.2f}%"
                )
                print(
                    f"Oracle ON : CE={s_on_or['ce']:.6f} "
                    f"Acc={s_on_or['acc']:.2f}%"
                )
                print(
                    "Oracle Seed effect "
                    f"(ON-OFF CE): {oracle_seed['mean']:+.8f} "
                    f"95%CI[{oracle_seed['ci95'][0]:+.8f},"
                    f"{oracle_seed['ci95'][1]:+.8f}]"
                )
                print(
                    f"Best fixed OFF: {off_key} | "
                    f"CE={s_off_best['ce']:.6f} "
                    f"Acc={s_off_best['acc']:.2f}%"
                )
                print(
                    f"Best fixed ON : {on_key} | "
                    f"CE={s_on_best['ce']:.6f} "
                    f"Acc={s_on_best['acc']:.2f}%"
                )

                report["budgets"][str(budget)] = {
                    "num_configs": len(configs),
                    "config_average_seed_on_minus_off": avg_seed,
                    "oracle_seed_on_minus_off": oracle_seed,
                    "oracle_off": s_off_or,
                    "oracle_on": s_on_or,
                    "best_fixed_off": {
                        "config": off_key,
                        **s_off_best,
                    },
                    "best_fixed_on": {
                        "config": on_key,
                        **s_on_best,
                    },
                }

                internal[budget] = {
                    "off_oracle": off_or,
                    "on_oracle": on_or,
                }

    print()
    print("=" * 100)
    print("CROSS-BUDGET: Mini seed가 Main head 하나를 일부 대체하는가?")
    print("=" * 100)

    for b in range(1, model.main_heads):
        low = internal[b]["on_oracle"]
        high = internal[b + 1]["off_oracle"]

        delta = bootstrap_ci(
            low["losses"] - high["losses"],
            args.bootstrap_repeats,
            args.seed + 3000 + b,
        )

        sl = summary(low)
        sh = summary(high)

        print(
            f"B={b} Seed ON Oracle "
            f"vs B={b+1} Seed OFF Oracle"
        )
        print(
            f"  B={b} ON : CE={sl['ce']:.6f}, "
            f"Acc={sl['acc']:.2f}%"
        )
        print(
            f"  B={b+1} OFF: CE={sh['ce']:.6f}, "
            f"Acc={sh['acc']:.2f}%"
        )
        print(
            f"  CE(lowON-highOFF)="
            f"{delta['mean']:+.8f} "
            f"95%CI[{delta['ci95'][0]:+.8f},"
            f"{delta['ci95'][1]:+.8f}]"
        )

        report["cross_budget"][
            f"B{b}_seedON_vs_B{b+1}_seedOFF"
        ] = {
            "low_seed_on": sl,
            "high_seed_off": sh,
            "ce_low_on_minus_high_off": delta,
        }

    print()
    print("=" * 100)
    print("AUTOMATIC CONSERVATIVE READ")
    print("=" * 100)

    clear = []
    for b in range(1, model.main_heads):
        ci = report["budgets"][str(b)][
            "config_average_seed_on_minus_off"
        ]["ci95"]
        if ci[1] < 0:
            clear.append(b)

    if clear:
        print(
            "SUPPORTED SIGNAL: constrained Main budget에서 "
            f"Seed ON이 Seed OFF보다 명확히 낮은 CE를 보인 budget={clear}."
        )
        print(
            "=> 현재 Q-seeding 경로가 Main 계산 일부를 보완할 가능성은 있다."
        )
        print(
            "단, 실제 FLOPs/latency 절감과 최종 논문 주장은 아직 아니다."
        )
    else:
        print(
            "WEAK SIGNAL: B=1/2의 config-average 비교에서 "
            "Seed ON 우위가 명확하지 않다."
        )
        print(
            "=> 현재 Q-seeding 방식만으로 'Mini가 필요한 Main head 수를 "
            "줄인다'는 가설의 증거는 약하다."
        )
        print(
            "다만 현재 모델에는 true Mini-only base output path가 없으므로 "
            "이 결과만으로 전체 연구 아이디어를 폐기하면 안 된다."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
