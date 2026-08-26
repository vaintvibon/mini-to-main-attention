import torch

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.balanced_direct_scheduler import BalancedDirectSubsetScheduler
from models.counterfactual_direct_utility import (
    CounterfactualDirectUtilityEvaluator,
)


torch.manual_seed(42)


MINI_HEADS = 4
DIRECT_K = 2
DEPTH = 2


# =============================================================
# 1. Small CIFAR-size model
# =============================================================

model = DynamicMiniMainViT(
    img_size=32,
    patch_size=4,
    num_classes=10,
    embed_dim=192,
    depth=DEPTH,
    main_heads=3,
    mini_heads=MINI_HEADS,
    mini_head_dim=16,
    pool_ratio=2,
    utility_hidden_dim=64,
    direct_k=DIRECT_K,
    mix_temperature=1.0,
    bind_dim=64,
    bind_temperature=1.0,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
)

model.eval()


# =============================================================
# 2. Balanced scheduler coverage
# =============================================================

scheduler = BalancedDirectSubsetScheduler(
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
)


print("Direct combinations:")
print(
    scheduler.combinations
)

assert scheduler.num_combinations == 6


# batch=2, 3 steps => block마다 총 6 samples => 모든 조합 1회씩
coverage = [
    []
    for _ in range(DEPTH)
]


for step in range(3):
    forced = scheduler.get_for_all_blocks(
        batch_size=2,
        depth=DEPTH,
        step=step,
    )

    for block_idx in range(
        DEPTH
    ):
        coverage[
            block_idx
        ].extend(
            [
                tuple(row)
                for row in forced[
                    block_idx
                ].tolist()
            ]
        )


for block_idx in range(
    DEPTH
):
    print(
        f"\nBlock {block_idx} coverage:"
    )
    print(
        coverage[
            block_idx
        ]
    )

    assert set(
        coverage[
            block_idx
        ]
    ) == set(
        scheduler.combinations
    )


# =============================================================
# 3. Forced routing forward
# =============================================================

x = torch.randn(
    2,
    3,
    32,
    32,
)

labels = torch.tensor(
    [2, 7],
    dtype=torch.long,
)


forced = scheduler.get_for_all_blocks(
    batch_size=2,
    depth=DEPTH,
    step=0,
)


logits, info_list = model(
    x,
    return_info=True,
    forced_direct_indices_per_block=forced,
    forced_uniform_mix=True,
)


print(
    "\nLogits shape:"
)
print(
    logits.shape
)


for block_idx, info in enumerate(
    info_list
):
    print(
        f"\nBlock {block_idx}"
    )

    print(
        "Forced Direct:"
    )
    print(
        forced[
            block_idx
        ]
    )

    print(
        "Actual Direct:"
    )
    print(
        info[
            "direct_indices"
        ]
    )

    print(
        "Mix weights:"
    )
    print(
        info[
            "mix_weights"
        ]
    )

    assert torch.equal(
        forced[
            block_idx
        ],
        info[
            "direct_indices"
        ],
    )

    assert info[
        "forced_routing"
    ] is True

    assert info[
        "forced_uniform_mix"
    ] is True

    # H=4, K=2 => remaining 2 heads.
    # forced_uniform_mix=True이므로 remaining weight는 각각 0.5.
    direct_weights = (
        info[
            "mix_weights"
        ]
        .masked_select(
            info[
                "direct_mask"
            ]
        )
    )

    remaining_weights = (
        info[
            "mix_weights"
        ]
        .masked_select(
            info[
                "remaining_mask"
            ]
        )
    )

    assert torch.allclose(
        direct_weights,
        torch.zeros_like(
            direct_weights
        ),
        atol=1e-7,
    )

    assert torch.allclose(
        remaining_weights,
        torch.full_like(
            remaining_weights,
            0.5,
        ),
        atol=1e-6,
    )


# =============================================================
# 4. Counterfactual Direct Utility Teacher
# =============================================================

evaluator = CounterfactualDirectUtilityEvaluator(
    model=model,
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
    target_temperature=1.0,
)


result = evaluator.evaluate(
    x,
    labels,
)


subset_losses = (
    result[
        "subset_losses"
    ]
)

head_utility = (
    result[
        "head_utility"
    ]
)

teacher_target = (
    result[
        "teacher_target"
    ]
)

best_subset = (
    result[
        "best_subset"
    ]
)


print(
    "\nSubset losses shape:"
)
print(
    subset_losses.shape
)


print(
    "\nHead Direct-inclusion utility:"
)
print(
    head_utility
)


print(
    "\nTeacher target:"
)
print(
    teacher_target
)


print(
    "\nTeacher target sum:"
)
print(
    teacher_target.sum(
        dim=-1
    )
)


print(
    "\nOracle best Direct subset:"
)
print(
    best_subset
)


# =============================================================
# 5. Invariants
# =============================================================

assert subset_losses.shape == (
    2,
    DEPTH,
    6,
)

assert head_utility.shape == (
    2,
    DEPTH,
    MINI_HEADS,
)

assert teacher_target.shape == (
    2,
    DEPTH,
    MINI_HEADS,
)

assert best_subset.shape == (
    2,
    DEPTH,
    DIRECT_K,
)

assert torch.isfinite(
    subset_losses
).all()

assert torch.isfinite(
    head_utility
).all()

assert torch.isfinite(
    teacher_target
).all()

assert torch.allclose(
    teacher_target.sum(dim=-1),
    torch.ones(
        2,
        DEPTH,
        dtype=teacher_target.dtype,
    ),
    atol=1e-6,
)

print(
    "\nBalanced routing + Counterfactual Utility test passed."
)
