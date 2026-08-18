import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT


torch.manual_seed(42)


# =============================================================
# Configuration
# =============================================================

STEPS = 30

LR = 1e-3

LAMBDA_UTILITY = 0.5


# =============================================================
# 1. Model
#
# sanity test이므로 32x32로 줄여서 빠르게 검사한다.
#
# CIFAR-10과 동일한 spatial size:
#
# 32x32
# patch=4
# -> 8x8 patches
# -> 64 patches + CLS
# -> 65 tokens
# =============================================================

model = DynamicMiniMainViT(
    img_size=32,
    patch_size=4,

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


model.train()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01,
)


# =============================================================
# 2. Fixed mini batch
#
# 실제 dataset 학습 전
# joint training mechanics만 검사.
# =============================================================

x = torch.randn(
    8,
    3,
    32,
    32,
)


labels = torch.tensor(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ],
    dtype=torch.long,
)


# =============================================================
# Helper
# =============================================================

def normalize_taylor(
    importance,
):

    importance_sum = (
        importance.sum(
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

    target = (
        importance
        /
        safe_sum
    )

    uniform = torch.full_like(
        target,
        1.0 / target.shape[-1],
    )

    target = torch.where(
        has_signal.expand_as(
            target
        ),
        target,
        uniform,
    )

    return target


def ranking_metrics(
    info_list,
    target,
):

    predicted_logits = torch.stack(
        [
            info[
                "utility_logits"
            ]
            for info in info_list
        ],
        dim=1,
    )

    # [B,depth,H]

    predicted_top1 = (
        predicted_logits.argmax(
            dim=-1
        )
    )

    teacher_top1 = (
        target.argmax(
            dim=-1
        )
    )

    top1 = (
        predicted_top1
        ==
        teacher_top1
    ).float().mean()


    predicted_top2 = torch.topk(
        predicted_logits,
        k=2,
        dim=-1,
    ).indices


    teacher_top2 = torch.topk(
        target,
        k=2,
        dim=-1,
    ).indices


    matches = (
        predicted_top2[
            :,
            :,
            :,
            None,
        ]
        ==
        teacher_top2[
            :,
            :,
            None,
            :,
        ]
    )

    # predicted Top2 각각이
    # teacher Top2 안에 존재하는지
    overlap = (
        matches
        .any(dim=-1)
        .float()
        .mean()
    )

    return (
        top1.item(),
        overlap.item(),
    )


# =============================================================
# Initial values
# =============================================================

first_task_loss = None
first_utility_loss = None

last_task_loss = None
last_utility_loss = None


# =============================================================
# 3. Joint training
# =============================================================

for step in range(
    1,
    STEPS + 1,
):

    optimizer.zero_grad(
        set_to_none=True
    )


    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    logits, info_list = model(
        x,
        return_info=True,
        collect_taylor=True,
    )


    # ---------------------------------------------------------
    # Task Loss
    # ---------------------------------------------------------

    sample_losses = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )


    task_loss = (
        sample_losses.mean()
    )


    # ---------------------------------------------------------
    # Taylor Teacher
    # ---------------------------------------------------------

    taylor_gates = [
        info[
            "taylor_gate"
        ]
        for info in info_list
    ]


    gate_grads = torch.autograd.grad(
        outputs=sample_losses.sum(),
        inputs=taylor_gates,

        create_graph=False,

        # 이후 total_loss.backward() 필요
        retain_graph=True,

        allow_unused=False,
    )


    taylor_importance = torch.stack(
        [
            grad.abs()
            for grad in gate_grads
        ],
        dim=1,
    )


    # ---------------------------------------------------------
    # Teacher target
    # ---------------------------------------------------------

    taylor_target = normalize_taylor(
        taylor_importance
    ).detach()


    # ---------------------------------------------------------
    # Utility Loss
    # ---------------------------------------------------------

    block_utility_losses = []


    for block_idx, info in enumerate(
        info_list
    ):

        utility_logits = (
            info[
                "utility_logits"
            ]
        )


        target = (
            taylor_target[
                :,
                block_idx,
                :,
            ]
        )


        block_loss = F.kl_div(
            F.log_softmax(
                utility_logits,
                dim=-1,
            ),

            target,

            reduction="batchmean",
        )


        block_utility_losses.append(
            block_loss
        )


    utility_loss = (
        torch.stack(
            block_utility_losses
        )
        .mean()
    )


    # ---------------------------------------------------------
    # Joint loss
    # ---------------------------------------------------------

    total_loss = (
        task_loss
        +
        LAMBDA_UTILITY
        *
        utility_loss
    )


    # ---------------------------------------------------------
    # Metrics before update
    # ---------------------------------------------------------

    (
        top1,
        top2,
    ) = ranking_metrics(
        info_list,
        taylor_target,
    )


    accuracy = (
        logits.argmax(dim=-1)
        ==
        labels
    ).float().mean().item()


    # raw Taylor magnitude도 기록
    #
    # 이 값이 지나치게 0으로 내려가면
    # normalized teacher가 noise에 민감해질 수 있다.
    mean_taylor = (
        taylor_importance
        .mean()
        .item()
    )


    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    total_loss.backward()


    # NaN / Inf 검사
    for name, parameter in (
        model.named_parameters()
    ):

        if parameter.grad is None:
            continue

        assert torch.isfinite(
            parameter.grad
        ).all(), (
            f"Non-finite gradient: {name}"
        )


    optimizer.step()


    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    if first_task_loss is None:

        first_task_loss = (
            task_loss.item()
        )

        first_utility_loss = (
            utility_loss.item()
        )


    last_task_loss = (
        task_loss.item()
    )

    last_utility_loss = (
        utility_loss.item()
    )


    if (
        step == 1
        or step % 5 == 0
        or step == STEPS
    ):

        print(
            f"\n"
            f"Step {step:02d}/{STEPS}"
        )

        print(
            f"Task loss: "
            f"{task_loss.item():.6f}"
        )

        print(
            f"Utility KL: "
            f"{utility_loss.item():.6f}"
        )

        print(
            f"Total loss: "
            f"{total_loss.item():.6f}"
        )

        print(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Taylor Top-1 agreement: "
            f"{top1 * 100:.2f}%"
        )

        print(
            f"Taylor Top-2 overlap: "
            f"{top2 * 100:.2f}%"
        )

        print(
            f"Mean raw Taylor: "
            f"{mean_taylor:.8e}"
        )


# =============================================================
# 4. Summary
# =============================================================

print(
    "\n"
    "================ SUMMARY ================"
)


print(
    "Task loss:"
)

print(
    first_task_loss,
    "->",
    last_task_loss,
)


print(
    "\nUtility KL:"
)

print(
    first_utility_loss,
    "->",
    last_utility_loss,
)


# =============================================================
# 5. Basic sanity checks
# =============================================================

assert (
    last_task_loss
    <
    first_task_loss
), (
    "Task loss did not decrease."
)


assert torch.isfinite(
    torch.tensor(
        last_utility_loss
    )
)


print(
    "\nDynamic Taylor joint overfit test passed."
)