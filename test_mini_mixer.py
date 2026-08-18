import torch

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility
from models.dynamic_mini_selector import DynamicMiniSelector
from models.mini_mixer import MiniMixer


torch.manual_seed(42)


# =============================================================
# 1. Multi Mini Attention
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
# 3. Dynamic Mini Selector
# =============================================================

selector = DynamicMiniSelector(
    mini_heads=4,
    direct_k=2,
)


# =============================================================
# 4. Mini Mixer
# =============================================================

mixer = MiniMixer(
    mini_heads=4,
    temperature=1.0,
)


# =============================================================
# 5. Dummy Input
# =============================================================

x = torch.randn(
    2,
    197,
    192,
)


# =============================================================
# 6. Multi Mini Attention
# =============================================================

mini_contexts, mini_attn = mini(
    x,
    patch_hw=(14, 14),
)


# =============================================================
# 7. Utility Prediction
# =============================================================

utility_logits, utility_info = utility(
    mini_contexts,
    mini_attn,
    return_info=True,
)


# =============================================================
# 8. Direct / Remaining Selection
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
# 9. Remaining Mini Mixing
# =============================================================

mixed_context, mix_info = mixer(
    mini_contexts,
    utility_logits,
    remaining_mask,
    return_info=True,
)


# =============================================================
# 10. 출력
# =============================================================

print("Utility logits:")
print(utility_logits)


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


print("\nMix weights:")
print(
    mix_info["mix_weights"]
)


print("\nMix weight sum:")
print(
    mix_info["mix_weight_sum"]
)


print("\nMixed context shape:")
print(
    mixed_context.shape
)


# =============================================================
# 11. 중요한 invariant 확인
# =============================================================

# Direct Head의 Mix weight는 반드시 0
direct_mix_weights = (
    mix_info["mix_weights"]
    .masked_select(direct_mask)
)

print(
    "\nDirect Head mix weights:"
)
print(
    direct_mix_weights
)


assert torch.allclose(
    direct_mix_weights,
    torch.zeros_like(
        direct_mix_weights
    ),
)


# Remaining이 있는 경우 weight 합은 1
assert torch.allclose(
    mix_info["mix_weight_sum"],
    torch.ones_like(
        mix_info["mix_weight_sum"]
    ),
    atol=1e-6,
)


print(
    "\nMiniMixer test passed."
)