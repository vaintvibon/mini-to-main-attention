import torch

from models.multi_mini_attention import MultiMiniAttention


model = MultiMiniAttention(
    dim=192,
    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,
)

x = torch.randn(
    2,
    197,
    192,
)

mini_contexts, mini_attn = model(
    x,
    patch_hw=(14, 14),
)

print(
    "Input:",
    x.shape,
)

print(
    "Mini contexts:",
    mini_contexts.shape,
)

print(
    "Mini attention:",
    mini_attn.shape,
)

for h in range(model.mini_heads):

    print(
        f"Mini Head {h}:",
        mini_contexts[:, h].shape,
    )