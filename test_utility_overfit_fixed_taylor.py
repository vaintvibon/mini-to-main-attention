import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT


torch.manual_seed(42)


# =============================================================
# Configuration
# =============================================================

STEPS = 50
LR = 5e-3


# =============================================================
# 1. Model
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


# deterministic sanity test
model.eval()


# =============================================================
# 2. Fixed batch
# =============================================================

x = torch.randn(
    4,
    3,
    224,
    224,
)

labels = torch.tensor(
    [3, 7, 1, 5],
    dtype=torch.long,
)


# =============================================================
# 3. Taylor Teacher를 딱 한 번 생성
# =============================================================

logits, info_list = model(
    x,
    return_info=True,
    collect_taylor=True,
)


sample_losses = F.cross_entropy(
    logits,
    labels,
    reduction="none",
)


taylor_gates = [
    info["taylor_gate"]
    for info in info_list
]


gate_grads = torch.autograd.grad(
    outputs=sample_losses.sum(),
    inputs=taylor_gates,

    create_graph=False,
    retain_graph=False,

    allow_unused=False,
)


# =============================================================
# 4. Taylor importance
#
# [B, depth, Hmini]
# =============================================================

taylor_importance = torch.stack(
    [
        grad.abs()
        for grad in gate_grads
    ],
    dim=1,
)


# =============================================================
# 5. Exact normalization
# =============================================================

importance_sum = (
    taylor_importance
    .sum(
        dim=-1,
        keepdim=True,
    )
)


has_signal = (
    importance_sum > 0
)


safe_sum = torch.where(
    has_signal,
    importance_sum,
    torch.ones_like(
        importance_sum
    ),
)


taylor_target = (
    taylor_importance
    /
    safe_sum
)


uniform_target = torch.full_like(
    taylor_target,
    1.0 / taylor_target.shape[-1],
)


taylor_target = torch.where(
    has_signal.expand_as(
        taylor_target
    ),
    taylor_target,
    uniform_target,
)


# Teacher는 gradient 없음
taylor_target = (
    taylor_target.detach()
)


print("Taylor target:")
print(taylor_target)


print(
    "\nTaylor target sum:"
)

print(
    taylor_target.sum(
        dim=-1
    )
)


# =============================================================
# 6. Mini representations도 고정
#
# 중요:
#
# Utility Predictor 자체가 Taylor target을 배울 수 있는지만
# 검사하기 위해 Mini feature도 cache한다.
#
# 따라서 이 테스트에서는 routing 변화가 끼어들지 않는다.
# =============================================================

cached_mini_contexts = [
    info["mini_contexts"]
    .detach()
    .clone()

    for info in info_list
]


cached_mini_attn = [
    info["mini_attn"]
    .detach()
    .clone()

    for info in info_list
]


# =============================================================
# 7. 모든 parameter freeze
# =============================================================

for parameter in model.parameters():
    parameter.requires_grad_(False)


# =============================================================
# Utility Predictor만 trainable
# =============================================================

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


optimizer = torch.optim.AdamW(
    utility_parameters,
    lr=LR,
    weight_decay=0.0,
)


# =============================================================
# Metric functions
# =============================================================

def get_metrics():
    """
    현재 Utility Predictor와
    고정 Taylor target 사이의 ranking agreement 계산.
    """

    all_logits = []


    with torch.no_grad():

        for block_idx, block in enumerate(
            model.blocks
        ):

            predictor = (
                block
                .attn
                .utility_predictor
            )

            utility_logits, _ = predictor(
                cached_mini_contexts[
                    block_idx
                ],
                cached_mini_attn[
                    block_idx
                ],
                return_info=True,
            )

            all_logits.append(
                utility_logits
            )


    # [B, depth, Hmini]
    predicted_logits = torch.stack(
        all_logits,
        dim=1,
    )


    # ---------------------------------------------------------
    # Top-1 agreement
    # ---------------------------------------------------------

    predicted_top1 = (
        predicted_logits
        .argmax(dim=-1)
    )

    teacher_top1 = (
        taylor_target
        .argmax(dim=-1)
    )


    top1_agreement = (
        predicted_top1
        ==
        teacher_top1
    ).float().mean()


    # ---------------------------------------------------------
    # Top-2 overlap
    #
    # Teacher Top2와 Predicted Top2 중
    # 몇 개가 겹치는지 / 2
    # ---------------------------------------------------------

    predicted_top2 = torch.topk(
        predicted_logits,
        k=2,
        dim=-1,
    ).indices


    teacher_top2 = torch.topk(
        taylor_target,
        k=2,
        dim=-1,
    ).indices


    overlaps = []


    B = predicted_logits.shape[0]
    depth = predicted_logits.shape[1]


    for b in range(B):

        for d in range(depth):

            pred_set = set(
                predicted_top2[
                    b,
                    d,
                ]
                .tolist()
            )

            target_set = set(
                teacher_top2[
                    b,
                    d,
                ]
                .tolist()
            )

            overlap = (
                len(
                    pred_set
                    &
                    target_set
                )
                / 2.0
            )

            overlaps.append(
                overlap
            )


    top2_overlap = (
        sum(overlaps)
        /
        len(overlaps)
    )


    return (
        predicted_logits,
        top1_agreement.item(),
        top2_overlap,
    )


# =============================================================
# 8. Before training
# =============================================================

(
    before_logits,
    before_top1,
    before_top2,
) = get_metrics()


print(
    "\n================ BEFORE ================"
)


print(
    "Predicted utility logits:"
)

print(
    before_logits
)


print(
    "\nTop-1 agreement:"
)

print(
    before_top1
)


print(
    "\nTop-2 overlap:"
)

print(
    before_top2
)


# =============================================================
# 9. Predictor-only overfit
# =============================================================

initial_loss = None
final_loss = None


for step in range(
    1,
    STEPS + 1,
):

    optimizer.zero_grad(
        set_to_none=True
    )


    block_losses = []


    for block_idx, block in enumerate(
        model.blocks
    ):

        predictor = (
            block
            .attn
            .utility_predictor
        )


        utility_logits, _ = predictor(
            cached_mini_contexts[
                block_idx
            ],
            cached_mini_attn[
                block_idx
            ],
            return_info=True,
        )


        target = (
            taylor_target[
                :,
                block_idx,
                :,
            ]
        )


        # =====================================================
        # KL divergence
        #
        # soft cross entropy와 predictor에 대한 gradient는
        # 동일한 방향이다.
        #
        # KL은 완벽히 target을 맞추면 0에 가까워져
        # sanity test 해석이 더 쉽다.
        # =====================================================

        block_loss = F.kl_div(
            F.log_softmax(
                utility_logits,
                dim=-1,
            ),

            target,

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


    if initial_loss is None:

        initial_loss = (
            utility_loss
            .detach()
            .item()
        )


    utility_loss.backward()


    optimizer.step()


    final_loss = (
        utility_loss
        .detach()
        .item()
    )


    if (
        step == 1
        or step % 10 == 0
        or step == STEPS
    ):

        (
            _,
            current_top1,
            current_top2,
        ) = get_metrics()


        print(
            f"\nStep {step:02d}/{STEPS}"
        )

        print(
            "KL loss:",
            final_loss,
        )

        print(
            "Top-1 agreement:",
            current_top1,
        )

        print(
            "Top-2 overlap:",
            current_top2,
        )


# =============================================================
# 10. Final evaluation
# =============================================================

(
    after_logits,
    after_top1,
    after_top2,
) = get_metrics()


print(
    "\n================ AFTER ================"
)


print(
    "Predicted utility logits:"
)

print(
    after_logits
)


print(
    "\nInitial KL loss:"
)

print(
    initial_loss
)


print(
    "\nFinal KL loss:"
)

print(
    final_loss
)


print(
    "\nTop-1 agreement:"
)

print(
    before_top1,
    "->",
    after_top1,
)


print(
    "\nTop-2 overlap:"
)

print(
    before_top2,
    "->",
    after_top2,
)


# =============================================================
# 11. Invariants
# =============================================================

assert final_loss < initial_loss, (
    "Utility Predictor failed to reduce "
    "Taylor-target KL loss."
)


assert after_top1 >= before_top1, (
    "Top-1 agreement became worse."
)


assert after_top2 >= before_top2, (
    "Top-2 overlap became worse."
)


assert torch.isfinite(
    after_logits
).all()


print(
    "\nFixed Taylor Utility overfit test passed."
)