import torch

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility
from models.dynamic_mini_selector import DynamicMiniSelector


torch.manual_seed(42)


# =============================================================
# 1. Mini Attention
# =============================================================

mini = MultiMiniAttention(
    dim=192,
    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,
)


# =============================================================
# 2. Utility Predictor
# =============================================================

utility = MiniHeadUtility(
    mini_head_dim=16,
    hidden_dim=64,
)


# =============================================================
# 3. Dynamic Selector
#
# 4개 Mini Head 중
# utility가 높은 2개를 Direct로 선택
# =============================================================

selector = DynamicMiniSelector(
    mini_heads=4,
    direct_k=2,
)


# =============================================================
# 4. Dummy input
# =============================================================

x = torch.randn(
    2,
    197,
    192,
)


# =============================================================
# 5. Mini forward
# =============================================================

mini_contexts, mini_attn = mini(
    x,
    patch_hw=(14, 14),
)


# =============================================================
# 6. Utility
# =============================================================

utility_logits, utility_info = utility(
    mini_contexts,
    mini_attn,
    return_info=True,
)


# =============================================================
# 7. Dynamic selection
# =============================================================

(
    direct_mask,
    remaining_mask,
    selection_info,
) = selector(
    utility_logits,
    return_info=True,
)


# =============================================================
# 8. 출력
# =============================================================

print("Utility logits:")
print(utility_logits)

print("\nRanking:")
print(
    torch.argsort(
        utility_logits,
        dim=-1,
        descending=True,
    )
)


print("\nDirect indices:")
print(
    selection_info["direct_indices"]
)


print("\nDirect mask:")
print(
    direct_mask
)


print("\nRemaining mask:")
print(
    remaining_mask
)


print("\nDirect count:")
print(
    selection_info["direct_count"]
)


print("\nRemaining count:")
print(
    selection_info["remaining_count"]
)