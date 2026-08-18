import torch

from models.dynamic_mini_main_block import (
    DynamicMiniMainBlock
)


torch.manual_seed(42)


# =============================================================
# 1. Block
# =============================================================

block = DynamicMiniMainBlock(
    dim=192,

    main_heads=3,

    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,

    utility_hidden_dim=64,

    direct_k=2,

    mix_temperature=1.0,

    bind_dim=64,
    bind_temperature=1.0,

    mlp_ratio=4.0,

    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
)


# =============================================================
# 2. Dummy ViT tokens
#
# CLS + 14x14 patches
#
# 1 + 196 = 197
# =============================================================

x = torch.randn(
    2,
    197,
    192,
)


# 원본 residual 비교용
x_before = x.clone()


# =============================================================
# 3. Forward
# =============================================================

out, info = block(
    x,
    patch_hw=(14, 14),
    return_info=True,
)


# =============================================================
# 4. Shape
# =============================================================

print("Input:")
print(
    x.shape
)


print("\nBlock output:")
print(
    out.shape
)


print("\nMini contexts:")
print(
    info["mini_contexts"].shape
)


print("\nUtility logits:")
print(
    info["utility_logits"]
)


print("\nDirect Mini indices:")
print(
    info["direct_indices"]
)


print("\nMix weights:")
print(
    info["mix_weights"]
)


print("\nBinding:")
print(
    info["binding_hard"]
)


print("\nMain attention:")
print(
    info["main_attn"].shape
)


# =============================================================
# 5. Routing 출력
# =============================================================

for b in range(
    x.shape[0]
):

    print(
        f"\n===== Sample {b} ====="
    )

    for mini_idx in (
        info["direct_indices"][b]
    ):

        mini_idx = (
            mini_idx.item()
        )

        main_indices = (
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

        assert len(
            main_indices
        ) == 1

        main_idx = (
            main_indices.item()
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
# 6. Basic invariants
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
    "main_attn"
].shape == (
    2,
    3,
    197,
    197,
)


# =============================================================
# 7. Block이 실제로 input을 변경했는지
# =============================================================

difference = (
    out
    -
    x_before
)

difference_norm = (
    difference
    .norm()
)


print(
    "\nBlock residual difference norm:"
)

print(
    difference_norm
)


assert (
    difference_norm.item()
    > 0.0
)


# =============================================================
# 8. NaN / Inf 검사
# =============================================================

assert torch.isfinite(
    out
).all()


print(
    "\nDynamicMiniMainBlock test passed."
)