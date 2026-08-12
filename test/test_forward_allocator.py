# test_forward_allocator.py

import torch

from models.mini_attention import MiniAttention
from models.mini_to_main_allocator import MiniImportance, MiniToMainAllocator


def main():
    torch.manual_seed(42)

    # DeiT-tiny scale
    B = 2
    D = 192
    H = 14
    W = 14
    N = 1 + H * W

    main_heads = 3

    x = torch.randn(B, N, D)

    mini_attn = MiniAttention(
        dim=D,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        has_cls_token=True,
    )

    mini_importance = MiniImportance()

    allocator = MiniToMainAllocator(
        dim=D,
        main_heads=main_heads,
        importance_dim=2,
        hidden_dim=128,
        dropout=0.0,
        use_cls_token=True,
    )

    c_m, attn = mini_attn(x, patch_hw=(H, W), return_attn=True)

    importance_feat, stats = mini_importance(attn)

    alloc_logits = allocator(c_m, importance_feat)

    print("Input x shape:           ", x.shape)
    print("Mini context shape:      ", c_m.shape)
    print("Mini attn shape:         ", attn.shape)
    print("Importance feature shape:", importance_feat.shape)
    print("Alloc logits shape:      ", alloc_logits.shape)

    print("\nImportance feature:")
    print(importance_feat)

    print("\nAlloc logits:")
    print(alloc_logits)

    print("\nStats:")
    for k, v in stats.items():
        print(f"{k}: {v.item():.6f}")

    assert c_m.shape == (B, N, D)
    assert attn.shape == (B, 1, N, 50)
    assert importance_feat.shape == (B, 2)
    assert alloc_logits.shape == (B, main_heads)

    print("\nMiniImportance + MiniToMainAllocator forward test passed.")


if __name__ == "__main__":
    main()