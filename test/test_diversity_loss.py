# test_diversity_loss.py

import random

import torch
import torch.nn as nn

from models.mini_guided_vit import MiniGuidedViT
from losses.diversity_loss import HeadDiversityLoss


def grad_norm(model: nn.Module) -> float:
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            norm = p.grad.detach().data.norm(2)
            total += norm.item() ** 2

    return total ** 0.5


def main():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    B = 2
    img_size = 224
    num_classes = 100
    budget_list = [0, 1, 2, 3]

    model = MiniGuidedViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=2,
        main_heads=3,
        mlp_ratio=4.0,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=128,
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    diversity_loss_fn = HeadDiversityLoss(
        mode="direct_mixed",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )

    lambda_div = 0.01

    model.train()

    x = torch.randn(B, 3, img_size, img_size).to(device)
    y = torch.randint(0, num_classes, (B,)).to(device)

    for budget in budget_list:
        optimizer.zero_grad(set_to_none=True)

        logits, info_list = model(
            x,
            budget=budget,
            return_info=True,
        )

        ce_loss = ce_loss_fn(logits, y)

        div_terms = []

        for info in info_list:
            div = diversity_loss_fn(
                head_out=info["head_out"],
                active_mask=info["active_mask"],
                direct_mask=info["direct_mask"],
                mixed_mask=info["mixed_mask"],
            )
            div_terms.append(div)

        div_loss = torch.stack(div_terms).mean()

        total_loss = ce_loss + lambda_div * div_loss

        total_loss.backward()
        norm = grad_norm(model)
        optimizer.step()

        print("\n" + "=" * 70)
        print(f"Budget: {budget}")
        print(f"CE loss:        {ce_loss.item():.6f}")
        print(f"Diversity loss: {div_loss.item():.6f}")
        print(f"Total loss:     {total_loss.item():.6f}")
        print(f"Grad norm:      {norm:.6f}")
        print("Logits shape:   ", logits.shape)

        assert logits.shape == (B, num_classes)
        assert torch.isfinite(total_loss).item()
        assert norm > 0.0

        if budget in [0, 1]:
            # budget 0: active head 없음
            # budget 1: direct만 있고 mixed가 없어서 direct_mixed diversity는 0
            assert div_loss.item() == 0.0

    print("\nDiversity loss test passed.")


if __name__ == "__main__":
    main()