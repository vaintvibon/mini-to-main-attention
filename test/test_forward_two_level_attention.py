# test_forward_two_level_attention.py

import torch

from models.two_level_attention import TwoLevelMiniMainAttention


def print_mask(name, mask):
    print(f"{name}:")
    print(mask.int())


def main():
    torch.manual_seed(42)

    # DeiT-tiny scale
    B = 2
    D = 192
    H = 14
    W = 14
    N = 1 + H * W

    main_heads = 3
    budget_list = [0, 1, 2, 3]

    x = torch.randn(B, N, D)

    attn = TwoLevelMiniMainAttention(
        dim=D,
        main_heads=main_heads,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        allocator_hidden_dim=128,
        has_cls_token=True,
    )

    for budget in budget_list:
        print("\n" + "=" * 70)
        print(f"Budget = {budget}")

        out, info = attn(
            x,
            budget=budget,
            patch_hw=(H, W),
            return_info=True,
        )

        print("Input x shape:          ", x.shape)
        print("Output shape:           ", out.shape)
        print("Mini context shape:     ", info["mini_context"].shape)
        print("Mini attn score shape:  ", info["mini_attn_score"].shape)
        print("Importance feat shape:  ", info["importance_feat"].shape)
        print("Alloc logits shape:     ", info["alloc_logits"].shape)
        print("Main out shape:         ", info["main_out"].shape)

        if budget > 0:
            print("Main attn shape:        ", info["main_attn"].shape)

        print("\nAlloc logits:")
        print(info["alloc_logits"])

        print()
        print_mask("active_mask", info["active_mask"])
        print_mask("direct_mask", info["direct_mask"])
        print_mask("mixed_mask", info["mixed_mask"])
        print_mask("inactive_mask", info["inactive_mask"])

        print("\nScheduler stats:")
        for k, v in info["scheduler_stats"].items():
            print(f"{k}: {v.item():.2f}")

        # shape checks
        assert out.shape == (B, N, D)
        assert info["mini_context"].shape == (B, N, D)
        assert info["mini_attn_score"].shape == (B, 1, N, 50)
        assert info["importance_feat"].shape == (B, 2)
        assert info["alloc_logits"].shape == (B, main_heads)
        assert info["main_out"].shape == (B, N, D)

        # mask checks
        assert info["active_mask"].shape == (B, main_heads)
        assert info["direct_mask"].shape == (B, main_heads)
        assert info["mixed_mask"].shape == (B, main_heads)
        assert info["inactive_mask"].shape == (B, main_heads)

        assert torch.all(info["active_mask"].float().sum(dim=1) == budget)
        assert torch.all(
            info["active_mask"] == (info["direct_mask"] | info["mixed_mask"])
        )
        assert torch.all(info["inactive_mask"] == (~info["active_mask"]))

        if budget == 0:
            assert torch.all(info["main_out"] == 0)
        else:
            assert info["main_attn"].shape == (B, main_heads, N, N)

    print("\nTwoLevelMiniMainAttention forward test passed.")


if __name__ == "__main__":
    main()