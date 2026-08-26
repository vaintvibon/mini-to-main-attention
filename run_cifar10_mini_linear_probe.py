# run_cifar10_mini_linear_probe.py
"""
보조 진단:
현재 Block-0 Multi-Mini representation이 Main Attention 전에
CIFAR-10 class 정보를 어느 정도 이미 담고 있는지 linear probe로 본다.

이 실험은 'Mini-only 모델 accuracy'가 아니다.
현재 Q-seeding-only 구조에는 true Mini-only output path가 없기 때문이다.

공식 CIFAR-10 test는 사용하지 않는다.
기본:
  probe train = seed42 permutation [11096,12096) 1000
  probe val   = seed42 permutation [12096,12596) 500
"""

import argparse
import math
import os
import random

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-start", type=int, default=11096)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-start", type=int, default=12096)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfgv(cfg, k, d):
    v = cfg.get(k, d)
    return d if v is None else v


def load_model(args, device):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    state = None
    for k in ("model", "state_dict", "model_state_dict"):
        if k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        raise KeyError("No model state in checkpoint.")

    cfg = ckpt.get("config", ckpt.get("args", {}))

    model = DynamicMiniMainViT(
        img_size=cfgv(cfg, "img_size", 32),
        patch_size=cfgv(cfg, "patch_size", 4),
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
        state, strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"missing={missing}\nunexpected={unexpected}"
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_subset(args, start, size, img_size):
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
    return Subset(base, perm[start:start + size])


@torch.no_grad()
def extract(model, ds, args, device):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    xs, ys = [], []
    for images, labels in loader:
        images = images.to(device)
        _, infos = model(images, return_info=True)

        mini = infos[0]["mini_contexts"]
        # [B,Hmini,N,Dmini] -> all Mini CLS -> [B,Hmini*Dmini]
        feat = mini[:, :, 0, :].flatten(1)

        xs.append(feat.cpu())
        ys.append(labels.cpu())

    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def eval_probe(probe, x, y):
    logits = probe(x)
    ce = F.cross_entropy(logits, y).item()
    acc = 100.0 * logits.argmax(-1).eq(y).float().mean().item()
    return ce, acc


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("=" * 88)
    print("MINI FEATURE LINEAR PROBE")
    print("=" * 88)
    print("Device:", device)
    print("Official CIFAR-10 test: NOT USED")

    model = load_model(args, device)
    img_size = getattr(model.patch_embed, "img_size", 32)

    tr = make_subset(
        args, args.train_start, args.train_size, img_size
    )
    va = make_subset(
        args, args.val_start, args.val_size, img_size
    )

    tx, ty = extract(model, tr, args, device)
    vx, vy = extract(model, va, args, device)

    mean = tx.mean(0, keepdim=True)
    std = tx.std(0, keepdim=True).clamp_min(1e-6)
    tx = ((tx - mean) / std).to(device)
    vx = ((vx - mean) / std).to(device)
    ty = ty.to(device)
    vy = vy.to(device)

    print("Feature shape:", tuple(tx.shape))
    print("Chance accuracy: 10.0%")

    probe = nn.Linear(tx.shape[1], 10).to(device)
    opt = torch.optim.AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_acc = -1.0
    best_ce = math.inf
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        probe.train()
        opt.zero_grad(set_to_none=True)
        logits = probe(tx)
        loss = F.cross_entropy(logits, ty)
        loss.backward()
        opt.step()

        probe.eval()
        vce, vacc = eval_probe(probe, vx, vy)

        if vacc > best_acc or (
            math.isclose(vacc, best_acc)
            and vce < best_ce
        ):
            best_acc = vacc
            best_ce = vce
            best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            tacc = (
                100.0
                * logits.argmax(-1).eq(ty).float().mean().item()
            )
            print(
                f"Epoch {epoch:03d} | "
                f"train CE={loss.item():.4f} Acc={tacc:.2f}% | "
                f"val CE={vce:.4f} Acc={vacc:.2f}%"
            )

    print()
    print(
        f"BEST VAL: CE={best_ce:.4f}, "
        f"Acc={best_acc:.2f}% @ epoch {best_epoch}"
    )
    print(
        "주의: 이는 Mini feature가 class 정보를 담는지 보는 probe이며, "
        "Mini-only 최종 모델 성능은 아니다."
    )


if __name__ == "__main__":
    main()
