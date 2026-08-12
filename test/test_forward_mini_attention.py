# test_forward_mini_attention.py

import torch

from models.mini_attention import MiniAttention


def main():
    torch.manual_seed(42)

    # DeiT-tiny scale
    B = 2
    D = 192
    H = 14
    W = 14
    N = 1 + H * W  # CLS token + patch tokens = 197

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

    c_m, attn = mini_attn(x, patch_hw=(H, W), return_attn=True)

    print("Input x shape:      ", x.shape)
    print("Mini context shape: ", c_m.shape)
    print("Mini attn shape:    ", attn.shape)

    expected_pooled_h = H // 2
    expected_pooled_w = W // 2
    expected_m = 1 + expected_pooled_h * expected_pooled_w

    assert c_m.shape == (B, N, D)
    assert attn.shape == (B, 1, N, expected_m)

    print("MiniAttention forward test passed.")


if __name__ == "__main__":
    main()