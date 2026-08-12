# test_train_step.py

import random
import torch
import torch.nn as nn

from models.mini_guided_vit import MiniGuidedViT


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm


def main():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 작은 sanity test용 설정
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
        depth=2,            # local sanity test이므로 2로 둠
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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )

    model.train()

    # 랜덤 이미지와 랜덤 라벨
    x = torch.randn(B, 3, img_size, img_size).to(device)
    y = torch.randint(0, num_classes, (B,)).to(device)

    for step in range(4):
        budget = random.choice(budget_list)

        optimizer.zero_grad(set_to_none=True)

        logits, info_list = model(
            x,
            budget=budget,
            return_info=True,
        )

        loss = criterion(logits, y)

        loss.backward()

        grad_norm = compute_grad_norm(model)

        optimizer.step()

        print("\n" + "=" * 70)
        print(f"Step: {step}")
        print(f"Budget: {budget}")
        print(f"Loss: {loss.item():.6f}")
        print(f"Grad norm: {grad_norm:.6f}")
        print("Logits shape:", logits.shape)

        # 첫 번째 block만 확인
        info = info_list[0]
        print("Block 0 active mask:")
        print(info["active_mask"].int())

        assert logits.shape == (B, num_classes)
        assert torch.isfinite(loss).item()
        assert grad_norm > 0.0

    print("\nOne-step training sanity test passed.")


if __name__ == "__main__":
    main()