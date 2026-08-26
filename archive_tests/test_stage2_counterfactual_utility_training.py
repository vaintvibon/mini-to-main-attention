import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.balanced_direct_scheduler import BalancedDirectSubsetScheduler
from models.counterfactual_direct_utility import (
    CounterfactualDirectUtilityEvaluator,
)


torch.manual_seed(42)


# =============================================================
# Configuration
# =============================================================

WARMUP_STEPS = 30
UTILITY_STEPS = 50

BACKBONE_LR = 1e-3
UTILITY_LR = 5e-3

BATCH_SIZE = 8

MINI_HEADS = 4
DIRECT_K = 2
DEPTH = 2


# =============================================================
# 1. Model
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


# =============================================================
# 2. Fixed sanity batch
# =============================================================

x = torch.randn(
    BATCH_SIZE,
    3,
    32,
    32,
)

labels = torch.arange(
    BATCH_SIZE,
    dtype=torch.long,
) % 10


# =============================================================
# 3. Balanced scheduler
# =============================================================

scheduler = BalancedDirectSubsetScheduler(
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
)


# =============================================================
# 4. Stage-1 warm-up
#
# Utility Predictor:
#   - freeze
#   - routing에 사용 X
#   - Mix에도 사용 X
#
# Backbone:
#   - Balanced Direct
#   - Uniform Remaining Mix
#   - CE only
# =============================================================

for block in model.blocks:
    predictor = (
        block
        .attn
        .utility_predictor
    )

    for parameter in predictor.parameters():
        parameter.requires_grad_(
            False
        )


backbone_optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=BACKBONE_LR,
    weight_decay=0.01,
)


model.train()

warmup_first_loss = None
warmup_last_loss = None


for step in range(
    WARMUP_STEPS
):
    backbone_optimizer.zero_grad(
        set_to_none=True
    )

    forced = scheduler.get_for_all_blocks(
        batch_size=BATCH_SIZE,
        depth=DEPTH,
        step=step,
        device=x.device,
    )

    logits = model(
        x,
        return_info=False,
        forced_direct_indices_per_block=forced,
        forced_uniform_mix=True,
    )

    loss = F.cross_entropy(
        logits,
        labels,
    )

    if warmup_first_loss is None:
        warmup_first_loss = (
            loss.detach().item()
        )

    loss.backward()
    backbone_optimizer.step()

    warmup_last_loss = (
        loss.detach().item()
    )


print(
    "================ STAGE 1 ================"
)

print(
    "Warm-up CE:"
)

print(
    warmup_first_loss,
    "->",
    warmup_last_loss,
)


# =============================================================
# 5. Stage-2 Counterfactual Teacher
#
# 같은 warm-up backbone을 고정한 채,
# 각 block에서 가능한 모든 Direct subset을 비교한다.
# =============================================================

model.eval()


evaluator = CounterfactualDirectUtilityEvaluator(
    model=model,
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
    target_temperature=1.0,
)


teacher_result = evaluator.evaluate(
    x,
    labels,
)


subset_losses = (
    teacher_result[
        "subset_losses"
    ]
)

head_utility = (
    teacher_result[
        "head_utility"
    ]
)

teacher_target = (
    teacher_result[
        "teacher_target"
    ]
)

oracle_best_subset = (
    teacher_result[
        "best_subset"
    ]
)

combination_table = (
    teacher_result[
        "combination_table"
    ]
)

reference_stacked = (
    teacher_result[
        "reference_direct_indices_per_block"
    ]
)

# [B, depth, K] -> list(depth) of [B,K]
reference_per_block = [
    reference_stacked[
        :,
        block_idx,
        :,
    ]
    for block_idx in range(DEPTH)
]


print(
    "\n================ STAGE 2 TEACHER ================"
)


print(
    "Subset losses shape:"
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
    "\nRaw utility mean abs:"
)

print(
    head_utility
    .abs()
    .mean()
    .item()
)


subset_spread = (
    subset_losses.max(dim=-1).values
    -
    subset_losses.min(dim=-1).values
)


print(
    "\nSubset loss spread per sample/block:"
)

print(
    subset_spread
)


print(
    "\nMean subset loss spread:"
)

print(
    subset_spread.mean().item()
)


print(
    "\nOracle best Direct subset:"
)

print(
    oracle_best_subset
)


# =============================================================
# Helper metrics
# =============================================================

def canonicalize_subset(
    subset: torch.Tensor,
):
    return (
        subset
        .sort(dim=-1)
        .values
    )


def exact_subset_match(
    predicted_subset: torch.Tensor,
    target_subset: torch.Tensor,
):
    predicted_subset = (
        canonicalize_subset(
            predicted_subset
        )
    )

    target_subset = (
        canonicalize_subset(
            target_subset
        )
    )

    return (
        predicted_subset
        ==
        target_subset
    ).all(dim=-1).float().mean().item()


def top2_overlap(
    predicted_subset: torch.Tensor,
    target_subset: torch.Tensor,
):
    matches = (
        predicted_subset[
            ...,
            :,
            None,
        ]
        ==
        target_subset[
            ...,
            None,
            :,
        ]
    )

    return (
        matches
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )


def top1_agreement(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    pred = (
        logits.argmax(
            dim=-1
        )
    )

    teacher = (
        target.argmax(
            dim=-1
        )
    )

    return (
        pred
        ==
        teacher
    ).float().mean().item()


def subset_to_combo_indices(
    subset: torch.Tensor,
    combo_table: torch.Tensor,
):
    """
    subset:
        [B, depth, K]

    combo_table:
        [num_combinations, K]

    return:
        [B, depth]
    """

    subset = (
        canonicalize_subset(
            subset
        )
    )

    combo_table = (
        canonicalize_subset(
            combo_table
        )
    )

    equality = (
        subset[
            :,
            :,
            None,
            :,
        ]
        ==
        combo_table[
            None,
            None,
            :,
            :,
        ]
    ).all(dim=-1)

    assert equality.any(
        dim=-1
    ).all(), (
        "At least one predicted subset was not found "
        "in the combination table."
    )

    return equality.float().argmax(
        dim=-1
    )


def subset_regret(
    selected_subset: torch.Tensor,
):
    combo_indices = (
        subset_to_combo_indices(
            selected_subset,
            combination_table,
        )
    )

    selected_loss = (
        subset_losses.gather(
            dim=-1,
            index=combo_indices[
                :,
                :,
                None,
            ],
        )
        .squeeze(-1)
    )

    oracle_loss = (
        subset_losses.min(
            dim=-1
        ).values
    )

    regret = (
        selected_loss
        -
        oracle_loss
    )

    return (
        regret.mean().item(),
        regret,
    )


# =============================================================
# 6. Check whether per-head teacher Top-K itself matches
#    the oracle pair.
#
# This is IMPORTANT:
# per-head utility can fail when pairwise interactions are strong.
# =============================================================

teacher_topk = torch.topk(
    teacher_target,
    k=DIRECT_K,
    dim=-1,
).indices


teacher_oracle_exact = (
    exact_subset_match(
        teacher_topk,
        oracle_best_subset,
    )
)


teacher_oracle_overlap = (
    top2_overlap(
        teacher_topk,
        oracle_best_subset,
    )
)


teacher_regret_mean, teacher_regret = (
    subset_regret(
        teacher_topk
    )
)


print(
    "\nTeacher Top-K vs Oracle exact pair match:"
)

print(
    teacher_oracle_exact
)


print(
    "\nTeacher Top-K vs Oracle Top-2 overlap:"
)

print(
    teacher_oracle_overlap
)


print(
    "\nTeacher Top-K mean oracle regret:"
)

print(
    teacher_regret_mean
)


print(
    "\nTeacher Top-K regret per sample/block:"
)

print(
    teacher_regret
)


# =============================================================
# 7. Freeze backbone completely.
#    Unfreeze Utility Predictor only.
# =============================================================

for parameter in model.parameters():
    parameter.requires_grad_(
        False
    )


utility_parameters = []


for block in model.blocks:
    predictor = (
        block
        .attn
        .utility_predictor
    )

    predictor.train()

    for parameter in predictor.parameters():
        parameter.requires_grad_(
            True
        )

        utility_parameters.append(
            parameter
        )


utility_optimizer = torch.optim.AdamW(
    utility_parameters,
    lr=UTILITY_LR,
    weight_decay=0.0,
)


# =============================================================
# 8. Predictor forward under fixed reference routing.
#
# Predictor is calculated,
# but routing is forced and remaining mix is uniform.
# Therefore the teacher does NOT move as predictor changes.
# =============================================================

def get_predicted_logits():
    _, info_list = model(
        x,
        return_info=True,
        collect_taylor=False,
        forced_direct_indices_per_block=reference_per_block,
        forced_uniform_mix=True,
    )

    return torch.stack(
        [
            info[
                "utility_logits"
            ]
            for info in info_list
        ],
        dim=1,
    )


with torch.no_grad():
    before_logits = (
        get_predicted_logits()
    )

    before_top1 = (
        top1_agreement(
            before_logits,
            teacher_target,
        )
    )

    before_topk = torch.topk(
        before_logits,
        k=DIRECT_K,
        dim=-1,
    ).indices

    before_teacher_exact = (
        exact_subset_match(
            before_topk,
            teacher_topk,
        )
    )

    before_oracle_exact = (
        exact_subset_match(
            before_topk,
            oracle_best_subset,
        )
    )

    before_regret_mean, _ = (
        subset_regret(
            before_topk
        )
    )


print(
    "\n================ BEFORE UTILITY TRAINING ================"
)


print(
    "Top-1 teacher agreement:"
)

print(
    before_top1
)


print(
    "\nPredicted Top-K vs Teacher exact pair match:"
)

print(
    before_teacher_exact
)


print(
    "\nPredicted Top-K vs Oracle exact pair match:"
)

print(
    before_oracle_exact
)


print(
    "\nPredicted mean oracle regret:"
)

print(
    before_regret_mean
)


# =============================================================
# 9. Train Utility Predictor on Counterfactual teacher
# =============================================================

initial_kl = None
final_kl = None


for step in range(
    1,
    UTILITY_STEPS + 1,
):
    utility_optimizer.zero_grad(
        set_to_none=True
    )

    predicted_logits = (
        get_predicted_logits()
    )

    block_losses = []

    for block_idx in range(
        DEPTH
    ):
        block_loss = F.kl_div(
            F.log_softmax(
                predicted_logits[
                    :,
                    block_idx,
                    :,
                ],
                dim=-1,
            ),
            teacher_target[
                :,
                block_idx,
                :,
            ],
            reduction="batchmean",
        )

        block_losses.append(
            block_loss
        )

    utility_loss = (
        torch.stack(
            block_losses
        )
        .mean()
    )

    if initial_kl is None:
        initial_kl = (
            utility_loss
            .detach()
            .item()
        )

    utility_loss.backward()

    for parameter in utility_parameters:
        if parameter.grad is None:
            continue

        assert torch.isfinite(
            parameter.grad
        ).all()

    utility_optimizer.step()

    final_kl = (
        utility_loss
        .detach()
        .item()
    )

    if (
        step == 1
        or step % 10 == 0
        or step == UTILITY_STEPS
    ):
        with torch.no_grad():
            current_logits = (
                get_predicted_logits()
            )

            current_top1 = (
                top1_agreement(
                    current_logits,
                    teacher_target,
                )
            )

            current_topk = torch.topk(
                current_logits,
                k=DIRECT_K,
                dim=-1,
            ).indices

            current_teacher_exact = (
                exact_subset_match(
                    current_topk,
                    teacher_topk,
                )
            )

            current_oracle_exact = (
                exact_subset_match(
                    current_topk,
                    oracle_best_subset,
                )
            )

            current_regret_mean, _ = (
                subset_regret(
                    current_topk
                )
            )

        print(
            f"\nStep {step:02d}/{UTILITY_STEPS}"
        )

        print(
            f"Utility KL: {final_kl:.6f}"
        )

        print(
            f"Top-1 teacher agreement: "
            f"{current_top1 * 100:.2f}%"
        )

        print(
            f"Pred Top-K vs Teacher exact: "
            f"{current_teacher_exact * 100:.2f}%"
        )

        print(
            f"Pred Top-K vs Oracle exact: "
            f"{current_oracle_exact * 100:.2f}%"
        )

        print(
            f"Pred mean oracle regret: "
            f"{current_regret_mean:.8f}"
        )


# =============================================================
# 10. Final evaluation
# =============================================================

with torch.no_grad():
    after_logits = (
        get_predicted_logits()
    )

    after_top1 = (
        top1_agreement(
            after_logits,
            teacher_target,
        )
    )

    after_topk = torch.topk(
        after_logits,
        k=DIRECT_K,
        dim=-1,
    ).indices

    after_teacher_exact = (
        exact_subset_match(
            after_topk,
            teacher_topk,
        )
    )

    after_oracle_exact = (
        exact_subset_match(
            after_topk,
            oracle_best_subset,
        )
    )

    after_regret_mean, after_regret = (
        subset_regret(
            after_topk
        )
    )


print(
    "\n================ SUMMARY ================"
)


print(
    "Utility KL:"
)

print(
    initial_kl,
    "->",
    final_kl,
)


print(
    "\nTop-1 teacher agreement:"
)

print(
    before_top1,
    "->",
    after_top1,
)


print(
    "\nPred Top-K vs Teacher exact pair:"
)

print(
    before_teacher_exact,
    "->",
    after_teacher_exact,
)


print(
    "\nPred Top-K vs Oracle exact pair:"
)

print(
    before_oracle_exact,
    "->",
    after_oracle_exact,
)


print(
    "\nPred mean oracle regret:"
)

print(
    before_regret_mean,
    "->",
    after_regret_mean,
)


print(
    "\nFinal regret per sample/block:"
)

print(
    after_regret
)


# =============================================================
# 11. Core assertions
#
# We require predictor learnability.
# We do NOT require teacher Top-K == oracle pair 100%.
# If that is low, it means pairwise interaction matters.
# =============================================================

assert final_kl < initial_kl, (
    "Utility Predictor failed to learn "
    "the Counterfactual teacher."
)


assert after_top1 >= before_top1, (
    "Top-1 teacher agreement became worse."
)


assert after_teacher_exact >= before_teacher_exact, (
    "Predicted Top-K moved away from the teacher Top-K."
)


assert torch.isfinite(
    after_logits
).all()


assert torch.isfinite(
    torch.tensor(
        after_regret_mean
    )
)


print(
    "\nStage-2 Counterfactual Utility training test passed."
)