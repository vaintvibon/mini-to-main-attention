# test_forward_block.py

import torch

from models.mini_guided_block import MiniGuidedBlock


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

    block = MiniGuidedBlock(
        dim=D,
        main_heads=main_heads,
        mlp_ratio=4.0,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        allocator_hidden_dim=128,
        has_cls_token=True,
    )

    for budget in budget_list:
        print("\n" + "=" * 70)
        print(f"Budget = {budget}")

        out, info = block(
            x,
            budget=budget,
            patch_hw=(H, W),
            return_info=True,
        )

        print("Input shape:        ", x.shape)
        print("Output shape:       ", out.shape)
        print("Mini context shape: ", info["mini_context"].shape)
        print("Alloc logits shape: ", info["alloc_logits"].shape)
        print("Main out shape:     ", info["main_out"].shape)

        if budget > 0:
            print("Main attn shape:    ", info["main_attn"].shape)

        print()
        print_mask("active_mask", info["active_mask"])
        print_mask("direct_mask", info["direct_mask"])
        print_mask("mixed_mask", info["mixed_mask"])
        print_mask("inactive_mask", info["inactive_mask"])

        assert out.shape == (B, N, D)
        assert info["mini_context"].shape == (B, N, D)
        assert info["alloc_logits"].shape == (B, main_heads)
        assert info["main_out"].shape == (B, N, D)

        assert torch.all(info["active_mask"].float().sum(dim=1) == budget)
        assert torch.all(
            info["active_mask"] == (info["direct_mask"] | info["mixed_mask"])
        )
        assert torch.all(info["inactive_mask"] == (~info["active_mask"]))

    print("\nMiniGuidedBlock forward test passed.")


if __name__ == "__main__":
    main()