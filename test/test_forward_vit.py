# test_forward_vit.py

import torch

from models.mini_guided_vit import MiniGuidedViT


def main():
    torch.manual_seed(42)

    B = 2
    img_size = 224
    num_classes = 100

    budget_list = [0, 1, 2, 3]

    # CPU에서도 빠르게 테스트하려고 depth=2로 둔다.
    # 실제 DeiT-tiny scale은 depth=12다.
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
    )

    x = torch.randn(B, 3, img_size, img_size)

    model.eval()

    with torch.no_grad():
        for budget in budget_list:
            print("\n" + "=" * 70)
            print(f"Budget = {budget}")

            logits, info_list = model(
                x,
                budget=budget,
                return_info=True,
            )

            print("Input image shape:", x.shape)
            print("Logits shape:     ", logits.shape)
            print("Num block infos:  ", len(info_list))

            assert logits.shape == (B, num_classes)
            assert len(info_list) == 2

            # 첫 번째 block 정보만 확인
            info = info_list[0]

            print("Block 0 mini context shape:", info["mini_context"].shape)
            print("Block 0 alloc logits shape:", info["alloc_logits"].shape)
            print("Block 0 active mask:")
            print(info["active_mask"].int())

            assert info["mini_context"].shape == (B, 197, 192)
            assert info["alloc_logits"].shape == (B, 3)
            assert torch.all(info["active_mask"].float().sum(dim=1) == budget)

    print("\nMiniGuidedViT forward test passed.")


if __name__ == "__main__":
    main()