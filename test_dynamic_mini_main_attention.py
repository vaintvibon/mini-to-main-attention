import torch

from models.dynamic_mini_main_attention import (
    DynamicMiniMainAttention
)


torch.manual_seed(42)


# =============================================================
# 1. Model
# =============================================================

model = DynamicMiniMainAttention(
    dim=192,

    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,

    utility_hidden_dim=64,

    direct_k=2,

    mix_temperature=1.0,

    main_heads=3,

    bind_dim=64,
    bind_temperature=1.0,
)


# =============================================================
# 2. Input
# =============================================================

x = torch.randn(
    2,
    197,
    192,
)


# =============================================================
# 3. Forward
# =============================================================

out, info = model(
    x,
    patch_hw=(14, 14),
    return_info=True,
)


# =============================================================
# 4. Basic shapes
# =============================================================

print("Input:")
print(x.shape)

print("\nOutput:")
print(out.shape)


print("\nMini contexts:")
print(
    info["mini_contexts"].shape
)


print("\nMini attention:")
print(
    info["mini_attn"].shape
)


print("\nUtility logits:")
print(
    info["utility_logits"]
)


print("\nUtility probabilities:")
print(
    info["utility_probs"]
)


print("\nDirect Mini indices:")
print(
    info["direct_indices"]
)


print("\nDirect mask:")
print(
    info["direct_mask"]
)


print("\nRemaining mask:")
print(
    info["remaining_mask"]
)


print("\nMix weights:")
print(
    info["mix_weights"]
)


print("\nBinding logits shape:")
print(
    info["binding_logits"].shape
)


print("\nHard binding:")
print(
    info["binding_hard"]
)


print("\nMain seeds:")
print(
    info["main_seeds"].shape
)


print("\nMain attention:")
print(
    info["main_attn"].shape
)


# =============================================================
# 5. Human-readable routing
# =============================================================

for b in range(
    x.shape[0]
):

    print(
        f"\n===== Sample {b} ====="
    )

    direct_indices = (
        info["direct_indices"][b]
    )

    for mini_idx in direct_indices:

        mini_idx = (
            mini_idx.item()
        )

        main_matches = (
            info[
                "binding_hard"
            ][
                b,
                mini_idx,
            ]
            .nonzero(
                as_tuple=False
            )
            .flatten()
        )

        assert (
            len(main_matches)
            == 1
        )

        main_idx = (
            main_matches
            .item()
        )

        print(
            f"Mini H{mini_idx} "
            f"-> Main H{main_idx} "
            "[DIRECT]"
        )

    mixed_main_indices = (
        info[
            "mixed_main_mask"
        ][b]
        .nonzero(
            as_tuple=False
        )
        .flatten()
        .tolist()
    )

    for main_idx in (
        mixed_main_indices
    ):

        print(
            "Remaining Mini Mix "
            f"-> Main H{main_idx} "
            "[MIX]"
        )


# =============================================================
# 6. Invariants
# =============================================================

assert out.shape == (
    2,
    197,
    192,
)


assert info[
    "mini_contexts"
].shape == (
    2,
    4,
    197,
    16,
)


assert info[
    "utility_logits"
].shape == (
    2,
    4,
)


assert info[
    "direct_mask"
].sum(
    dim=-1
).tolist() == [
    2,
    2,
]


assert info[
    "remaining_mask"
].sum(
    dim=-1
).tolist() == [
    2,
    2,
]


assert info[
    "binding_logits"
].shape == (
    2,
    4,
    3,
)


assert info[
    "main_seeds"
].shape == (
    2,
    3,
    197,
    64,
)


assert info[
    "main_attn"
].shape == (
    2,
    3,
    197,
    197,
)


# =============================================================
# 7. Direct Head는 Mix에 들어가면 안 됨
# =============================================================

direct_mix_weights = (
    info["mix_weights"]
    .masked_select(
        info["direct_mask"]
    )
)

assert torch.allclose(
    direct_mix_weights,
    torch.zeros_like(
        direct_mix_weights
    ),
)


# =============================================================
# 8. Remaining Mix weight sum = 1
# =============================================================

assert torch.allclose(
    info[
        "mix_weights"
    ].sum(dim=-1),
    torch.ones(
        x.shape[0]
    ),
    atol=1e-6,
)


# =============================================================
# 9. Direct Mini 하나 = Main 하나
# =============================================================

direct_binding_count = (
    info[
        "binding_hard"
    ]
    .sum(
        dim=(1, 2)
    )
)

assert torch.all(
    direct_binding_count
    == 2
)


# =============================================================
# 10. 하나의 Main에는 최대 Direct Mini 하나
# =============================================================

incoming = (
    info[
        "binding_hard"
    ]
    .sum(dim=1)
)

assert torch.all(
    incoming <= 1
)


print(
    "\nDynamicMiniMainAttention "
    "test passed."
)