import torch

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility


torch.manual_seed(42)


mini = MultiMiniAttention(
    dim=192,
    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,
)


utility = MiniHeadUtility(
    mini_head_dim=16,
    hidden_dim=64,
)


x = torch.randn(
    2,
    197,
    192,
)


mini_contexts, mini_attn = mini(
    x,
    patch_hw=(14, 14),
)


utility_logits, info = utility(
    mini_contexts,
    mini_attn,
    return_info=True,
)


print("Mini contexts:")
print(mini_contexts.shape)

print("\nMini attention:")
print(mini_attn.shape)

print("\nUtility logits:")
print(utility_logits.shape)

print("\nAttention entropy:")
print(info["attention_entropy"])

print("\nMax confidence:")
print(info["max_confidence"])

print("\nUtility logits:")
print(utility_logits)

print("\nUtility probabilities:")
print(info["utility_probs"])

print("\nRanking:")
print(
    torch.argsort(
        utility_logits,
        dim=-1,
        descending=True,
    )
)