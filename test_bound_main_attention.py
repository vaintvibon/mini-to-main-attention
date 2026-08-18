import torch

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility
from models.dynamic_mini_selector import DynamicMiniSelector
from models.mini_mixer import MiniMixer
from models.mini_main_binder import MiniMainBinder
from models.bound_main_attention import BoundMainAttention


torch.manual_seed(42)


# =============================================================
# Configuration
# =============================================================

DIM = 192

MINI_HEADS = 4
MINI_HEAD_DIM = 16

MAIN_HEADS = 3

# 192 / 3 = 64
MAIN_HEAD_DIM = 64

DIRECT_K = 2


# =============================================================
# 1. Multi Mini Attention
# =============================================================

mini = MultiMiniAttention(
    dim=DIM,
    mini_heads=MINI_HEADS,
    mini_head_dim=MINI_HEAD_DIM,
    pool_ratio=2,
)


# =============================================================
# 2. Utility Predictor
# =============================================================

utility = MiniHeadUtility(
    mini_head_dim=MINI_HEAD_DIM,
    hidden_dim=64,
)


# =============================================================
# 3. Dynamic Mini Selector
# =============================================================

selector = DynamicMiniSelector(
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
)


# =============================================================
# 4. Remaining Mini Mixer
# =============================================================

mixer = MiniMixer(
    mini_heads=MINI_HEADS,
    temperature=1.0,
)


# =============================================================
# 5. Dynamic Mini -> Main Binder
# =============================================================

binder = MiniMainBinder(
    mini_head_dim=MINI_HEAD_DIM,
    main_heads=MAIN_HEADS,
    main_head_dim=MAIN_HEAD_DIM,
    bind_dim=64,
)


# =============================================================
# 6. Main Attention
# =============================================================

main_attention = BoundMainAttention(
    dim=DIM,
    main_heads=MAIN_HEADS,
)


# =============================================================
# 7. Input
# =============================================================

x = torch.randn(
    2,
    197,
    DIM,
)


# =============================================================
# 8. Mini
# =============================================================

mini_contexts, mini_attn = mini(
    x,
    patch_hw=(14, 14),
)


# =============================================================
# 9. Utility
# =============================================================

utility_logits, utility_info = utility(
    mini_contexts,
    mini_attn,
    return_info=True,
)


# =============================================================
# 10. Selection
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
# 11. Mix
# =============================================================

mixed_context, mix_info = mixer(
    mini_contexts,
    utility_logits,
    remaining_mask,
    return_info=True,
)


# =============================================================
# 12. Dynamic Binding
# =============================================================

main_seeds, bind_info = binder(
    mini_contexts,
    selection_info["direct_indices"],
    mixed_context,
    return_info=True,
)


# =============================================================
# 13. Main Attention
# =============================================================

main_out, main_info = main_attention(
    x,
    main_seeds,
    return_info=True,
)


# =============================================================
# 14. 출력
# =============================================================

print("Input:")
print(
    x.shape
)


print("\nMini contexts:")
print(
    mini_contexts.shape
)


print("\nUtility logits:")
print(
    utility_logits
)


print("\nDirect Mini indices:")
print(
    selection_info[
        "direct_indices"
    ]
)


print("\nMix weights:")
print(
    mix_info[
        "mix_weights"
    ]
)


print("\nHard Mini -> Main binding:")
print(
    bind_info[
        "binding_hard"
    ]
)


print("\nMain seeds:")
print(
    main_seeds.shape
)


print("\nSeeded Main Q:")
print(
    main_info[
        "seeded_q"
    ].shape
)


print("\nMain attention:")
print(
    main_info[
        "main_attn"
    ].shape
)


print("\nMain Head outputs:")
print(
    main_info[
        "head_out"
    ].shape
)


print("\nFinal Main output:")
print(
    main_out.shape
)


# =============================================================
# 15. 사람이 읽기 쉬운 Binding 출력
# =============================================================

for b in range(
    x.shape[0]
):

    print(
        f"\n===== Sample {b} ====="
    )

    for mini_idx in (
        selection_info[
            "direct_indices"
        ][b]
    ):

        mini_idx = (
            mini_idx.item()
        )

        main_idx = (
            bind_info[
                "binding_hard"
            ][
                b,
                mini_idx,
            ]
            .nonzero(
                as_tuple=False
            )
            .item()
        )

        print(
            f"Mini H{mini_idx} "
            f"-> Main H{main_idx} "
            "[DIRECT]"
        )

    mixed_main_indices = (
        bind_info[
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
# 16. Shape invariants
# =============================================================

assert main_seeds.shape == (
    2,
    MAIN_HEADS,
    197,
    MAIN_HEAD_DIM,
)


assert main_info[
    "seeded_q"
].shape == (
    2,
    MAIN_HEADS,
    197,
    MAIN_HEAD_DIM,
)


assert main_info[
    "main_attn"
].shape == (
    2,
    MAIN_HEADS,
    197,
    197,
)


assert main_info[
    "head_out"
].shape == (
    2,
    MAIN_HEADS,
    197,
    MAIN_HEAD_DIM,
)


assert main_out.shape == (
    2,
    197,
    DIM,
)


# =============================================================
# 17. Seed가 실제 Q에 들어갔는지 확인
# =============================================================

q_difference = (
    main_info["seeded_q"]
    -
    main_info["q_base"]
)

difference_norm = (
    q_difference
    .norm()
)

print(
    "\nQ seed difference norm:"
)

print(
    difference_norm
)


assert (
    difference_norm.item()
    > 0.0
)


print(
    "\nBoundMainAttention test passed."
)