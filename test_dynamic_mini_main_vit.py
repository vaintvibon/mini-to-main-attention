import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import (
    DynamicMiniMainViT
)


torch.manual_seed(42)


# =============================================================
# 1. Model
#
# 우선 기존 테스트와 연결하기 위해
# 224x224 / patch16을 유지한다.
#
# depth=2만 사용해서 빠르게 sanity check.
# =============================================================

model = DynamicMiniMainViT(
    img_size=224,
    patch_size=16,

    num_classes=10,

    embed_dim=192,
    depth=2,

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

    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
)


# =============================================================
# 2. Dummy Images
# =============================================================

x = torch.randn(
    2,
    3,
    224,
    224,
)


labels = torch.tensor(
    [
        3,
        7,
    ],
    dtype=torch.long,
)


# =============================================================
# 3. Forward
# =============================================================

logits, info_list = model(
    x,
    return_info=True,
)


print("Input:")
print(
    x.shape
)


print("\nLogits:")
print(
    logits.shape
)


print("\nLogits values:")
print(
    logits
)


print("\nNumber of blocks:")
print(
    len(info_list)
)


# =============================================================
# 4. Block별 routing 확인
# =============================================================

for block_idx, info in enumerate(
    info_list
):

    print(
        f"\n"
        f"========================================"
    )

    print(
        f"Block {block_idx}"
    )

    print(
        f"========================================"
    )


    print(
        "\nUtility logits:"
    )

    print(
        info[
            "utility_logits"
        ]
    )


    print(
        "\nDirect Mini indices:"
    )

    print(
        info[
            "direct_indices"
        ]
    )


    print(
        "\nMix weights:"
    )

    print(
        info[
            "mix_weights"
        ]
    )


    # ---------------------------------------------------------
    # Sample별 Mini -> Main 연결
    # ---------------------------------------------------------

    for b in range(
        x.shape[0]
    ):

        print(
            f"\nSample {b}:"
        )

        for mini_idx in (
            info[
                "direct_indices"
            ][b]
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

            assert (
                len(main_indices)
                == 1
            )

            main_idx = (
                main_indices.item()
            )

            print(
                f"  Mini H{mini_idx} "
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
                "  Remaining Mini Mix "
                f"-> Main H{main_idx} "
                "[MIX]"
            )


# =============================================================
# 5. Classification Loss
#
# 이제 처음으로 실제 task loss가 생긴다.
# =============================================================

loss = F.cross_entropy(
    logits,
    labels,
)


print(
    "\nClassification loss:"
)

print(
    loss
)


# =============================================================
# 6. Backward
# =============================================================

model.zero_grad()

loss.backward()


print(
    "\nBackward completed."
)


# =============================================================
# 7. Gradient sanity check
# =============================================================

classifier_grad = (
    model.head.weight.grad
)


assert (
    classifier_grad
    is not None
)


assert torch.isfinite(
    classifier_grad
).all()


print(
    "\nClassifier grad norm:"
)

print(
    classifier_grad.norm()
)


# =============================================================
# 8. Mini Attention gradient 확인
# =============================================================

mini_q_grad = (
    model.blocks[0]
    .attn
    .mini_attention
    .q_proj
    .weight
    .grad
)


assert (
    mini_q_grad
    is not None
)


print(
    "\nBlock 0 Mini Q grad norm:"
)

print(
    mini_q_grad.norm()
)


# =============================================================
# 9. Utility Predictor gradient 확인
#
# 현재 완전한 Taylor supervision은 아직 없지만,
# Mix weight 경로를 통해 일부 gradient가 들어오는지 확인.
# =============================================================

utility_grad = (
    model.blocks[0]
    .attn
    .utility_predictor
    .scorer[-1]
    .weight
    .grad
)


print(
    "\nBlock 0 Utility Predictor grad:"
)

if utility_grad is None:

    print(
        "None"
    )

else:

    print(
        utility_grad.norm()
    )


# =============================================================
# 10. Shape invariants
# =============================================================

assert logits.shape == (
    2,
    10,
)


assert len(
    info_list
) == 2


for info in info_list:

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
# 11. Final finite check
# =============================================================

assert torch.isfinite(
    logits
).all()


assert torch.isfinite(
    loss
)


print(
    "\nDynamicMiniMainViT test passed."
)