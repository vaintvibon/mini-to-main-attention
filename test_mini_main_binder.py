import torch

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility
from models.dynamic_mini_selector import DynamicMiniSelector
from models.mini_mixer import MiniMixer
from models.mini_main_binder import MiniMainBinder


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
# 2. Utility
# =============================================================

utility = MiniHeadUtility(
    mini_head_dim=16,
    hidden_dim=64,
)


# =============================================================
# 3. Selector
#
# 4 Mini 중 2개 Direct
# =============================================================

selector = DynamicMiniSelector(
    mini_heads=4,
    direct_k=2,
)


# =============================================================
# 4. Mixer
# =============================================================

mixer = MiniMixer(
    mini_heads=4,
    temperature=1.0,
)


# =============================================================
# 5. Dynamic Mini -> Main Binder
#
# Main:
# 3 heads
#
# embed_dim=192라면
# main_head_dim=64
# =============================================================

binder = MiniMainBinder(
    mini_head_dim=16,
    main_heads=3,
    main_head_dim=64,
    bind_dim=64,
    temperature=1.0,
)


# =============================================================
# 6. Input
# =============================================================

x = torch.randn(
    2,
    197,
    192,
)


# =============================================================
# 7. Mini
# =============================================================

mini_contexts, mini_attn = mini(
    x,
    patch_hw=(14, 14),
)


# =============================================================
# 8. Utility
# =============================================================

utility_logits, utility_info = utility(
    mini_contexts,
    mini_attn,
    return_info=True,
)


# =============================================================
# 9. Selector
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
# 10. Mixer
# =============================================================

mixed_context, mix_info = mixer(
    mini_contexts,
    utility_logits,
    remaining_mask,
    return_info=True,
)


# =============================================================
# 11. Binder
# =============================================================

main_seeds, bind_info = binder(
    mini_contexts,
    selection_info["direct_indices"],
    mixed_context,
    return_info=True,
)


# =============================================================
# 12. 출력
# =============================================================

print("Utility logits:")
print(utility_logits)


print("\nDirect Mini indices:")
print(
    selection_info["direct_indices"]
)


print("\nMix weights:")
print(
    mix_info["mix_weights"]
)


print("\nBinding logits shape:")
print(
    bind_info["binding_logits"].shape
)


print("\nBinding logits:")
print(
    bind_info["binding_logits"]
)


print("\nHard binding matrix:")
print(
    bind_info["binding_hard"]
)


print("\nBound Main mask:")
print(
    bind_info["bound_main_mask"]
)


print("\nMixed Main mask:")
print(
    bind_info["mixed_main_mask"]
)


print("\nIncoming Direct count per Main:")
print(
    bind_info["incoming_direct_count"]
)


print("\nMain seeds shape:")
print(
    main_seeds.shape
)


# =============================================================
# 13. 사람이 읽기 쉽게 binding 출력
# =============================================================

B = x.shape[0]

for b in range(B):

    print(
        f"\n===== Sample {b} ====="
    )

    direct_indices = (
        selection_info[
            "direct_indices"
        ][b]
    )

    for mini_idx in direct_indices:

        mini_idx = mini_idx.item()

        main_idx = (
            bind_info[
                "binding_hard"
            ][b, mini_idx]
            .nonzero(
                as_tuple=False
            )
            .item()
        )

        print(
            f"Mini H{mini_idx} "
            f"-> Main H{main_idx} "
            f"[DIRECT]"
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

    for main_idx in mixed_main_indices:

        print(
            f"Mixed Remaining Mini "
            f"-> Main H{main_idx} "
            f"[MIX]"
        )


# =============================================================
# 14. Invariant tests
# =============================================================

# main seed shape
assert main_seeds.shape == (
    2,
    3,
    197,
    64,
)


# Direct Mini = 2이므로
# 각 sample에서 bound Main도 정확히 2개
assert torch.all(
    bind_info[
        "bound_main_mask"
    ].sum(dim=-1)
    == 2
)


# Main 3개 중 Direct 2개이므로
# Mixed Main은 정확히 1개
assert torch.all(
    bind_info[
        "mixed_main_mask"
    ].sum(dim=-1)
    == 1
)


# 한 Main에 Direct Mini가 둘 이상 들어가면 안 됨
assert torch.all(
    bind_info[
        "incoming_direct_count"
    ]
    <= 1
)


print(
    "\nMiniMainBinder test passed."
)