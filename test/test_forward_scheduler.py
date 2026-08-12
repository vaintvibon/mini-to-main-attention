# test_forward_scheduler.py

import torch

from models.mini_attention import MiniAttention
from models.mini_to_main_allocator import MiniImportance, MiniToMainAllocator
from models.two_level_scheduler import TwoLevelHeadScheduler


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

    scheduler = TwoLevelHeadScheduler(
        main_heads=main_heads,
        direct_ratio=0.34,
    )

    c_m, attn = mini_attn(x, patch_hw=(H, W), return_attn=True)
    importance_feat, importance_stats = mini_importance(attn)
    alloc_logits = allocator(c_m, importance_feat)

    print("Input x shape:           ", x.shape)
    print("Mini context shape:      ", c_m.shape)
    print("Mini attn shape:         ", attn.shape)
    print("Importance feature shape:", importance_feat.shape)
    print("Alloc logits shape:      ", alloc_logits.shape)

    print("\nAlloc logits:")
    print(alloc_logits)

    for budget in budget_list:
        print("\n" + "=" * 60)
        print(f"Budget = {budget}")

        out = scheduler(alloc_logits, budget=budget)

        active_mask = out["active_mask"]
        direct_mask = out["direct_mask"]
        mixed_mask = out["mixed_mask"]
        inactive_mask = out["inactive_mask"]
        stats = out["stats"]

        print_mask("active_mask", active_mask)
        print_mask("direct_mask", direct_mask)
        print_mask("mixed_mask", mixed_mask)
        print_mask("inactive_mask", inactive_mask)

        print("Stats:")
        for k, v in stats.items():
            print(f"{k}: {v.item():.2f}")

        # shape checks
        assert active_mask.shape == (B, main_heads)
        assert direct_mask.shape == (B, main_heads)
        assert mixed_mask.shape == (B, main_heads)
        assert inactive_mask.shape == (B, main_heads)

        # count checks
        assert torch.all(active_mask.float().sum(dim=1) == budget)

        if budget == 0:
            assert torch.all(direct_mask.float().sum(dim=1) == 0)
            assert torch.all(mixed_mask.float().sum(dim=1) == 0)
        else:
            # direct는 최소 1개
            assert torch.all(direct_mask.float().sum(dim=1) >= 1)
            # direct는 active subset
            assert torch.all((direct_mask & inactive_mask) == 0)

        # active = direct + mixed
        assert torch.all(active_mask == (direct_mask | mixed_mask))

        # inactive = not active
        assert torch.all(inactive_mask == (~active_mask))

    print("\nTwoLevelHeadScheduler forward test passed.")


if __name__ == "__main__":
    main()